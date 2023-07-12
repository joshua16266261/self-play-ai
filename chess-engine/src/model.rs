use std::cmp::min;

// use crate::tictactoe::{Policy, State};
use crate::game::{Policy, State, Encoding};
use crate::mcts::Args;
use tch::{Tensor, IndexOp, Device, Reduction, nn::{Adam, OptimizerConfig, VarStore, SequentialT, ModuleT, FuncT}, data::Iter2, kind::Kind};
use tch::nn;
use indicatif::ProgressBar;

// TODO: Separate Model and TicTacToeNet

pub struct TicTacToeNet {
    torso: SequentialT,
    policy_head: SequentialT,
    value_head: SequentialT
}

pub struct Model {
    pub args: Args,
    pub net: TicTacToeNet
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

fn resnet(vs: &nn::Path, num_resnet_blocks: u32, in_channels: i64, num_hidden: i64) -> SequentialT {
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

impl TicTacToeNet {
    pub fn new(vs: &nn::Path, num_resnet_blocks: u32, num_hidden: i64) -> TicTacToeNet {
        let conv_config = nn::ConvConfig { padding: 1, ..Default::default() };
        TicTacToeNet {
            torso: nn::seq_t()
                .add_fn(|x| x.view((-1, 3, 3, 3)))
                .add(resnet(vs, num_resnet_blocks, 3, num_hidden)),
            policy_head: nn::seq_t()
                .add(nn::conv2d(vs, num_hidden, 32, 3, conv_config))
                .add(nn::batch_norm2d(vs, 32, Default::default()))
                .add_fn(|x| x.relu())
                .add_fn(|x| x.flat_view())
                .add(nn::linear(vs, 32 * 9, 9, Default::default())),
            value_head: nn::seq_t()
                .add(nn::conv2d(vs, num_hidden, 3, 3, conv_config))
                .add(nn::batch_norm2d(vs, 3, Default::default()))
                .add_fn(|x| x.relu())
                .add_fn(|x| x.flat_view())
                .add(nn::linear(vs, 3 * 9, 1, Default::default()))
                .add_fn(|x| x.tanh())
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let torso_output = self.torso.forward_t(x, train);
        let policy = self.policy_head.forward_t(&torso_output, train);
        let value = self.value_head.forward_t(&torso_output, train);

        (policy, value)
    }
}

impl Model {
    pub fn predict<T: State>(&mut self, state: &T) -> (T::Policy, f32) {
        let encoded_state = state.encode();
        let encoded_state_slice = encoded_state.get_flat_slice();
        // let valid_actions = state.get_valid_actions();
        
        let input = Tensor::from_slice(encoded_state_slice)
            // .view(T::Encoding::get_original_shape())
            // .unsqueeze(0)
            .to(Device::Mps);
        let (mut policy, value) = self.net.forward(&input, false);
        policy = policy.softmax(-1, Kind::Float);

        let policy_vec = Vec::<f32>::try_from(policy.contiguous().view(-1)).unwrap();
        let value_float = value.double_value(&[0]) as f32;

        let masked_policy = state.mask_invalid_actions(policy_vec).unwrap();

        (masked_policy, value_float)
    }

    pub fn train<T: State>(
        &mut self,
        states: Vec<T::Encoding>,
        policies: Vec<T::Policy>,
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


