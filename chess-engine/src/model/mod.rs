pub mod tictactoe;

use tch::{
    Tensor,
    Device,
    Reduction,
    IndexOp,
    kind::Kind,
    data::Iter2,
    nn,
    nn::{VarStore, Adam, OptimizerConfig, SequentialT, FuncT}
};
use indicatif::ProgressBar;
use std::cmp::min;

use crate::game::{Policy, State, Encoding};
// use crate::mcts::Args;
use crate::mcts_parallel::Args;

pub trait Net {
    type State: State;

    fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor);
}

pub struct Model<T: Net> {
    pub args: Args,
    pub net: T
}

impl<T: Net> Model<T> {
    // pub fn predict(&mut self, state: &T::State) -> (<<T as crate::model::Net>::State as State>::Policy, f32) {
    //     let encoded_state = state.encode();
    //     let encoded_state_slice = encoded_state.get_flat_slice();
        
    //     let input = Tensor::from_slice(encoded_state_slice)
    //         .to(Device::Mps);
    //     let (mut policy, value) = self.net.forward(&input, false);
    //     policy = policy.softmax(-1, Kind::Float);

    //     let policy_vec = Vec::<f32>::try_from(policy.contiguous().view(-1)).unwrap();
    //     let value_float = value.double_value(&[0]) as f32;

    //     let masked_policy = state.mask_invalid_actions(policy_vec).unwrap();

    //     (masked_policy, value_float)
    // }

    pub fn predict(&mut self, states: &Vec<&T::State>) -> (Vec<<<T as crate::model::Net>::State as State>::Policy>, Vec<f32>) {
        let mut all_encoded_states: Vec<f32> = Vec::new();
        for state in states {
            all_encoded_states.extend_from_slice(state.encode().get_flat_slice());
        }
        
        let input = Tensor::from_slice(&all_encoded_states).to(Device::Mps);
        let (mut policy_tensor, value_tensor) = self.net.forward(&input, false);
        policy_tensor = policy_tensor.softmax(-1, Kind::Float);

        let mut policies: Vec<<<T as crate::model::Net>::State as State>::Policy> = Vec::with_capacity(states.len());
        for i in 0..states.len() {
            let state = states.get(i).unwrap();
            let policy_vec = Vec::<f32>::try_from(policy_tensor.i((i as i64, ..)).contiguous().view(-1)).unwrap();
            let masked_policy = state.mask_invalid_actions(policy_vec).unwrap();
            policies.push(masked_policy);
        }
        // let value_float = value.double_value(&[0]) as f32;

        let values = Vec::<f32>::try_from(value_tensor.contiguous().view(-1)).unwrap();

        (policies, values)
    }

    pub fn train(
        &mut self,
        states: Vec<<<T as crate::model::Net>::State as State>::Encoding>,
        policies: Vec<<<T as crate::model::Net>::State as State>::Policy>,
        values: Vec<f32>,
        var_store: &VarStore,
        args: Args,
        pb: &ProgressBar) {

        let mut optimizer = Adam::default().build(var_store, 1e-3).unwrap();

        let num_inputs = states.len() as i64;

        // Convert states to tensor of shape (num_inputs, size_of_flatten_state)
        let mut states_flattened: Vec<f32> = Vec::new();
        for state in states {
            states_flattened.extend_from_slice(state.get_flat_slice());
        }
        let state_tensors = Tensor::from_slice(&states_flattened).view((num_inputs, -1));

        // Convert policies to tensor of shape (num_inputs, size_of_policy)
        let mut policies_flattened: Vec<f32> = Vec::new();
        for policy in policies {
            policies_flattened.extend_from_slice(policy.get_flat_slice());
        }
        let policy_tensors = Tensor::from_slice(&policies_flattened).view((num_inputs, -1));

        // Convert values to tensor of shape (num_inputs, 1)
        let value_tensors = Tensor::from_slice(&values).view((num_inputs, 1)).to(Device::Mps);

        pb.reset();
        for _ in 0..args.num_epochs {
            let mut idx = 0;
            let mut loss = 0.0;

            for (state_batch, policy_batch) in
                Iter2::new(&state_tensors, &policy_tensors, args.batch_size)
                    .shuffle()
                    .to_device(Device::Mps)
                    .return_smaller_last_batch() {

                let value_batch = value_tensors.i(idx..min(idx+args.batch_size, num_inputs));

                let (policy, value) = self.net.forward(&state_batch, true);

                let policy_softmax = policy.log_softmax(-1, Kind::Float);
                let policy_nll = policy_softmax * policy_batch;
                let policy_loss = -policy_nll.sum(Kind::Float) / num_inputs;

                let value_loss = value.mse_loss(&value_batch, Reduction::Mean);
                let total_loss = policy_loss + value_loss;

                optimizer.backward_step(&total_loss);

                idx += args.batch_size;
                loss = total_loss.double_value(&[]);
            }
            pb.set_message(format!("Loss: {:.3}", loss));
            pb.inc(1);
        }
        pb.finish();
    }
}

fn resnet_block<'a>(vs: &nn::Path, num_hidden: i64) -> FuncT<'a> {
    let conv2d_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
    let seq = nn::seq_t()
        .add(nn::conv2d(vs, num_hidden, num_hidden, 3, conv2d_cfg))
        .add(nn::batch_norm2d(vs, num_hidden, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::conv2d(vs, num_hidden, num_hidden, 3, conv2d_cfg))
        .add(nn::batch_norm2d(vs, num_hidden, Default::default()));

    nn::func_t(move |x, train| {
        let f_x = x.apply_t(&seq, train);
        (x + f_x).relu()
    })
}

pub fn new_resnet(vs: &nn::Path, num_resnet_blocks: u32, in_channels: i64, num_hidden: i64) -> SequentialT {
    let mut net = nn::seq_t()
        .add(nn::conv2d(
            vs,
            in_channels,
            num_hidden,
            3,
            nn::ConvConfig { padding: 1, ..Default::default() }
        ))
        .add(nn::batch_norm2d(vs, num_hidden, Default::default()))
        .add_fn(|x| x.relu());

    for _ in 0..num_resnet_blocks {
        net = net.add(resnet_block(vs, num_hidden));
    }

    net
}
