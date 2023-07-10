use std::cmp::min;

use crate::tictactoe::{Policy, State};
use tch::{Tensor, IndexOp, TrainableCModule, IValue, Device, Reduction, nn::{Adam, OptimizerConfig, VarStore}, data::Iter2, kind::Kind};
use indicatif::ProgressBar;

pub struct Model {
    pub net: TrainableCModule
}

impl Model {
    pub fn predict(&mut self, state: &State) -> (Policy, f32) {
        // TODO: Don't set_eval here
        self.net.set_eval();

        let encoded_state = state.encode();
        let valid_actions = state.get_valid_actions();

        let input = Tensor::from_slice(&encoded_state).view((3, 3, 3)).to_device(Device::Mps);
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
        let mut total_prob = 0.0;
        for action in valid_actions {
            let idx = action.row * 3 + action.col;
            let prob = *policy_vec.get(idx).unwrap();
            return_policy[idx] = prob;
            total_prob += prob;
        }
        for prob in return_policy.iter_mut() {
            *prob /= total_prob;
        }

        (return_policy, value_float)
    }

    pub fn train(
        &mut self,
        states: Vec<[f32; 27]>,
        policies: Vec<Policy>,
        values: Vec<f32>,
        var_store: &VarStore,
        batch_size: i64,
        num_epochs: u32,
        pb: &ProgressBar) {
        self.net.set_train();

        let mut optimizer = Adam::default().build(var_store, 1e-4).unwrap();

        let num_inputs = states.len() as i64;

        // Convert states to tensor of shape (num_inputs, 27)
        let mut states_flattened: Vec<f32> = Vec::new();
        for state in states {
            states_flattened.extend_from_slice(&state);
        }
        let state_tensors = Tensor::from_slice(&states_flattened).view((num_inputs, 27));

        // Convert policies to tensor of shape (num_inputs, 9)
        let mut policies_flattened: Vec<f32> = Vec::new();
        for policy in policies {
            policies_flattened.extend_from_slice(&policy);
        }
        let policy_tensors = Tensor::from_slice(&policies_flattened).view((num_inputs, 9));

        // Convert values to tensor of shape (num_inputs, 1)
        let mut value_tensors = Tensor::from_slice(&values).to(Device::Mps);
        value_tensors = value_tensors.view((num_inputs, 1));

        pb.reset();
        for _ in 0..num_epochs {
            let mut idx = 0;
            let mut loss = 0.0;
            for (state_batch, policy_batch) in
                Iter2::new(&state_tensors, &policy_tensors, batch_size).shuffle().to_device(Device::Mps).return_smaller_last_batch() {
                let value_batch = value_tensors.i(idx..min(idx+batch_size, num_inputs));

                let output = self.net.forward_is(&[IValue::from(state_batch)]).unwrap();
                let (policy, value) = match output {
                    IValue::Tuple(ivalues) => match &ivalues[..] {
                        [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
                        _ => panic!("unexpected output {:?}", ivalues),
                    },
                    _ => panic!("unexpected output {:?}", output),
                };

                let policy_softmax = policy.log_softmax(-1, Kind::Float);
                let policy_nll = policy_softmax * policy_batch;
                let policy_loss = -policy_nll.sum(Kind::Float) / num_inputs;
                let value_loss = value.mse_loss(&value_batch, Reduction::Mean);
                let total_loss = policy_loss + value_loss;

                optimizer.backward_step(&total_loss);

                idx += batch_size;
                loss = total_loss.double_value(&[]);
            }
            pb.set_message(format!("Loss: {:.3}", loss));
            pb.inc(1);
        }
        pb.finish();
    }
}