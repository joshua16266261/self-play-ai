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
use ndarray::{stack, Axis, ArrayView, Ix3, Array2, ArrayD, Array1, Array4};
use rayon::prelude::*;

use crate::game::{Policy, State};
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
    pub fn predict(&self, states: &Vec<&T::State>) -> (Vec<<<T as crate::model::Net>::State as State>::Policy>, Vec<f32>) {
        // TODO: Parallelize
        // let mut all_encoded_states: Vec<f32> = Vec::new();
        // for state in states {
        //     all_encoded_states.extend_from_slice(state.get_encoding().get_flat_slice());
        // }

        // let all_encoded_states = states
        //     .par_iter()
        //     .fold(
        //         Vec::new,
        //         |mut acc: Vec<f32>, x| {
        //             acc.extend_from_slice(x.encode().get_flat_slice());
        //             acc
        //         }
        //     )
        //     .reduce(
        //         Vec::new,
        //         |mut acc, mut x| {
        //             acc.append(&mut x);
        //             acc
        //         }, 
        //     );

        // let a = states
        //     .iter()
        //     .map(|state| state.get_encoding().view())
        //     .collect::<Vec<_>>()
        //     .as_slice();

        let encoded_states = states
            .par_iter()
            .map(|state| state.get_encoding())
            .collect::<Vec<_>>();

        let encoded_state_views = encoded_states
            .par_iter()
            .map(|state| state.view());

        let encoded_states_ndarray = stack(
            Axis(0),
            encoded_state_views
                .collect::<Vec<_>>()
                .as_slice()
        ).unwrap();

        // let encoded_states = stack(
        //     Axis(0),
        //     states
        //         .iter()
        //         .map(|state| state.get_encoding().view())
        //         .collect::<Vec<_>>()
        //         .as_slice()
        // ).unwrap();

        let input = Tensor::try_from(encoded_states_ndarray).unwrap().to(Device::Mps);
        // println!("{}", input.to(Device::Cpu));
        // println!("{}", Tensor::from_slice(&[1, 2, 3, 4, 5, 6]));
        // let input = Tensor::from_slice(&all_encoded_states).to(Device::Mps);

        let (mut policy_tensor, value_tensor) = self.net.forward(&input, false);
        policy_tensor = policy_tensor.softmax(-1, Kind::Float);
        let policy_tensor_shape = policy_tensor.size2().unwrap();

        // println!("{}", policy_tensor.to(Device::Cpu));

        let policy_ndarray: ArrayD<f32> = (&policy_tensor).try_into().unwrap();
        let policy_ndarray = policy_ndarray.into_shape((policy_tensor_shape.0 as usize, policy_tensor_shape.1 as usize)).unwrap();

        // TODO: Parallelize
        let mut policies: Vec<<<T as crate::model::Net>::State as State>::Policy> = Vec::with_capacity(states.len());
        for i in 0..states.len() {
            let state = states.get(i).unwrap();

            // let policy_vec = Vec::<f32>::try_from(policy_tensor.i((i as i64, ..)).contiguous().view(-1)).unwrap();
            // let masked_policy = state.mask_invalid_actions(policy_vec).unwrap();
            let policy = policy_ndarray.index_axis(Axis(0), i);
            
            let masked_policy = state.mask_invalid_actions(policy).unwrap();
            policies.push(masked_policy);
        }

        // let policy_vec = Vec::<f32>::try_from(policy_tensor.contiguous().view(-1)).unwrap();

        // let a = states
        //     .par_iter()
        //     .enumerate()
        //     .fold(
        //         Vec::new,
        //         |mut acc, (i, state)| {
        //             let policy_vec = Vec::<f32>::try_from(policy_tensor.i((i as i64, ..)).contiguous().view(-1)).unwrap();
        //             let masked_policy = state.mask_invalid_actions(policy_vec).unwrap();
        //             acc.push(masked_policy);
        //             acc
        //         }
        //     );

        let values = Vec::<f32>::try_from(value_tensor.contiguous().view(-1)).unwrap();

        (policies, values)
    }

    pub fn train(
        &self,
        states: Tensor,
        policies: Tensor,
        values: Tensor,
        var_store: &VarStore,
        args: Args,
        pb: &ProgressBar) {

        let mut optimizer = Adam::default().build(var_store, 1e-3).unwrap();

        let num_inputs = states.size()[0];
        let num_batches = (num_inputs as f32 / self.args.batch_size as f32).ceil() as i64;
        let index = Tensor::randperm(num_inputs, (Kind::Int64, Device::Mps));
        
        let states = states.index_select(0, &index);
        let policies = policies.index_select(0, &index);
        let values = values.index_select(0, &index);

        // Convert states to tensor of shape (num_inputs, size_of_flatten_state)
        // let mut states_flattened: Vec<f32> = Vec::new();
        // for state in encoded_states {
        //     states_flattened.extend_from_slice(state.get_flat_slice());
        // }
        // let state_tensors = Tensor::from_slice(&states_flattened).view((num_inputs, -1));

        // let state_tensors = Tensor::try_from(encoded_states).unwrap().to(Device::Mps);

        // Convert policies to tensor of shape (num_inputs, size_of_policy)
        // let mut policies_flattened: Vec<f32> = Vec::new();
        // for policy in policies {
        //     policies_flattened.extend_from_slice(policy.get_flat_slice());
        // }
        // let policy_tensors = Tensor::from_slice(&policies_flattened).view((num_inputs, -1));

        // let policy_tensors = Tensor::try_from(policies).unwrap().to(Device::Mps);

        // Convert values to tensor of shape (num_inputs, 1)

        // let value_tensors = Tensor::from_slice(&values).view((num_inputs, 1)).to(Device::Mps);

        pb.reset();

        for _ in 0..args.num_epochs {
            let mut loss = 0.0;

            // for (state_batch, policy_batch) in
            //     Iter2::new(&states, &policies, args.batch_size)
            //         .shuffle()
            //         .to_device(Device::Mps)
            //         .return_smaller_last_batch() {

            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * args.batch_size;
                let end_idx = min(start_idx + args.batch_size, num_inputs);

                let state_batch = states.i(start_idx..end_idx);
                let policy_batch = policies.i(start_idx..end_idx);
                let value_batch = values.i(start_idx..end_idx);

                let (policy, value) = self.net.forward(&state_batch, true);

                let policy_softmax = policy.log_softmax(-1, Kind::Float);
                let policy_nll = policy_softmax * policy_batch;
                let policy_loss = -policy_nll.sum(Kind::Float) / num_inputs;

                let value_loss = value.mse_loss(&value_batch, Reduction::Mean);
                let total_loss = policy_loss + value_loss;

                optimizer.backward_step(&total_loss);

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
