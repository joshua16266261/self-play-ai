use std::{iter::zip, cmp::min, num};

use crate::tictactoe::Policy;
use tch::{Tensor, IndexOp, TrainableCModule, IValue, Device, Reduction, nn::{Adam, OptimizerConfig, VarStore}, data::Iter2};

pub struct Model {
    pub net: TrainableCModule
}

impl Model {
    pub fn predict(&mut self, state: [f32; 27]) -> (Policy, f32) {
        // TODO: Don't set_eval here
        self.net.set_eval();

        let input = Tensor::from_slice(&state).view((3, 3, 3)).to_device(Device::Mps);
        let output = self.net.forward_is(&[IValue::from(input.unsqueeze(0))]).unwrap();

        let (policy, value) = match output {
            IValue::Tuple(ivalues) => match &ivalues[..] {
                [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
                _ => panic!("unexpected output {:?}", ivalues),
            },
            _ => panic!("unexpected output {:?}", output),
        };

        let policy_vec = Vec::<f32>::try_from(policy.view(-1)).unwrap();
        let value_float = value.double_value(&[0]) as f32;

        let mut return_policy = [0f32; 9];
        for (i, prob) in return_policy.iter_mut().enumerate() {
            *prob = *policy_vec.get(i).unwrap();
        }

        (return_policy, value_float)
    }

    pub fn train(&mut self, states: Vec<[f32; 27]>, policies: Vec<Policy>, values: Vec<f32>, varstore: &VarStore, batch_size: i64) {
        self.net.set_train();

        let mut optimizer = Adam::default().build(&varstore, 1e-4).unwrap();

        let num_inputs = states.len() as i64;

        // Convert states to tensor of shape (num_inputs, 27)
        let mut states_flattened: Vec<f32> = Vec::new();
        for state in states {
            states_flattened.extend_from_slice(&state);
        }
        let state_tensors = Tensor::from_slice(&states_flattened).view((num_inputs, 27));
        // let state_tensors: Vec<Tensor> = states.iter().map(|x| Tensor::from_slice(x)).collect();
        // let state_batches = state_tensors.chunks(batch_size);

        // Convert policies to tensor of shape (num_inputs, 9)
        let mut policies_flattened: Vec<f32> = Vec::new();
        for policy in policies {
            policies_flattened.extend_from_slice(&policy);
        }
        let policy_tensors = Tensor::from_slice(&policies_flattened).view((num_inputs, 27)).view((num_inputs, 9));
        // let policy_tensors: Vec<Tensor> = policies.iter().map(|x| Tensor::from_slice(x)).collect();
        // let policy_batches = policy_tensors.chunks(batch_size);

        // Convert values to tensor of shape (num_inputs, 1)
        let mut value_tensors = Tensor::from_slice(&values).to(Device::Mps);
        // value_tensors = value_tensors.view((num_inputs, 1));
        // let value_tensors: Vec<Tensor> = values.iter().map(|x| Tensor::from_slice(&[*x])).collect();
        // let value_batches = value_tensors.chunks(batch_size);

        let batch_iter = Iter2::new(&state_tensors, &policy_tensors, batch_size).shuffle().to_device(Device::Mps);

        // for ((state_batch, policy_batch), value_batch) in
        //     zip(zip(state_batches, policy_batches), value_batches) {
        let mut idx = 0;
        for (state_batch, policy_batch) in batch_iter {
            let value_batch = value_tensors.i(idx..min(idx+batch_size, num_inputs));

            let output = self.net.forward_is(&[IValue::from(state_batch)]).unwrap();
            let (policy, value) = match output {
                IValue::Tuple(ivalues) => match &ivalues[..] {
                    [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
                    _ => panic!("unexpected output {:?}", ivalues),
                },
                _ => panic!("unexpected output {:?}", output),
            };

            let policy_loss = policy.cross_entropy_for_logits(&policy_batch);
            let value_loss = value.mse_loss(&value_batch, Reduction::Mean);
            let total_loss = policy_loss + value_loss;

            optimizer.backward_step(&total_loss);

            println!("Loss: {:.3}", total_loss.double_value(&[0]));
        }
    }
}