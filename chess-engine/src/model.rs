use std::cmp::min;

use crate::tictactoe::{Policy, State};
use crate::mcts::Args;
use tch::{Tensor, IndexOp, TrainableCModule, IValue, Device, Reduction, nn::{Adam, OptimizerConfig, VarStore, SequentialT, ModuleT, FuncT}, data::Iter2, kind::Kind};
use tch::nn;
use indicatif::ProgressBar;

pub struct TicTacToeNet {
    torso: SequentialT,
    policy_head: SequentialT,
    value_head: SequentialT
}

pub struct Model {
    // pub net: TrainableCModule
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
            torso: resnet(vs, num_resnet_blocks, 3, num_hidden),
            policy_head: nn::seq_t()
                .add(nn::conv2d(vs, num_hidden, 32, 3, conv_config))
                .add(nn::batch_norm2d(vs, 32, Default::default()))
                .add_fn(|x| x.relu())
                .add_fn(|x| x.flat_view())
                .add(nn::linear(vs, 32 * 9, 9, Default::default())),
                // .add_fn(|x| x.log_softmax(-1, Kind::Float)),
            value_head: nn::seq_t()
                .add(nn::conv2d(vs, num_hidden, 3, 3, conv_config))
                .add(nn::batch_norm2d(vs, 3, Default::default()))
                .add_fn(|x| x.relu())
                .add_fn(|x| x.flat_view())
                .add(nn::linear(vs, 3 * 9, 1, Default::default()))
                .add_fn(|x| x.tanh())
        }
    }

    // pub fn new(vs: &nn::Path, num_resnet_blocks: u32, num_hidden: i64) -> TicTacToeNet {
    //     TicTacToeNet {
    //         torso: nn::seq_t()
    //             .add_fn(|x| x.flat_view())
    //             .add(nn::linear(vs, 27, 64, Default::default()))
    //             .add_fn(|x| x.relu())
    //             .add(nn::linear(vs, 64, 32, Default::default()))
    //             .add_fn(|x| x.relu()),
    //         policy_head: nn::seq_t()
    //             .add(nn::linear(vs, 32, 9, Default::default()))
    //             .add_fn(|x| x.log_softmax(-1, Kind::Float)),
    //         value_head: nn::seq_t()
    //             .add(nn::linear(vs, 32, 8, Default::default()))
    //             .add_fn(|x| x.relu())
    //             .add(nn::linear(vs, 8, 1, Default::default()))
    //             // .add_fn(|x| x.tanh())
    //     }
    // }

    fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let torso_output = self.torso.forward_t(x, train);
        let policy = self.policy_head.forward_t(&torso_output, train);
        let value = self.value_head.forward_t(&torso_output, train);

        (policy, value)
    }
}

impl Model {
    pub fn predict(&mut self, state: &State) -> (Policy, f32) {
        // TODO: Don't set_eval here
        // self.net.set_eval();

        let encoded_state = state.encode();
        let valid_actions = state.get_valid_actions();

        let input = Tensor::from_slice(&encoded_state).view((1, 3, 3, 3)).to(Device::Mps);
        // let output = self.net.forward_is(&[IValue::from(input.unsqueeze(0))]).unwrap();
        let (policy, value) = self.net.forward(&input, false);

        // let (policy, value) = match output {
        //     IValue::Tuple(ivalues) => match &ivalues[..] {
        //         [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
        //         _ => panic!("unexpected output {:?}", ivalues),
        //     },
        //     _ => panic!("unexpected output {:?}", output),
        // };

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
        args: Args,
        pb: &ProgressBar) {
        // self.net.set_train();

        let mut optimizer = Adam::default().build(var_store, 1e-3).unwrap();

        let num_inputs = states.len() as i64;

        // Convert states to tensor of shape (num_inputs, 27)
        let mut states_flattened: Vec<f32> = Vec::new();
        for state in states {
            states_flattened.extend_from_slice(&state);
        }
        let state_tensors = Tensor::from_slice(&states_flattened).view((num_inputs, 3, 3, 3));

        // Convert policies to tensor of shape (num_inputs, 9)
        let mut policies_flattened: Vec<f32> = Vec::new();
        for policy in policies {
            policies_flattened.extend_from_slice(&policy);
        }
        let policy_tensors = Tensor::from_slice(&policies_flattened).view((num_inputs, 9));

        // Convert values to tensor of shape (num_inputs, 1)
        let mut value_tensors = Tensor::from_slice(&values);
        value_tensors = value_tensors.view((num_inputs, 1)).to(Device::Mps);

        pb.reset();
        for _ in 0..args.num_epochs {
            let mut idx = 0;
            let mut loss = 0.0;
            for (state_batch, policy_batch) in
                Iter2::new(&state_tensors, &policy_tensors, args.batch_size).shuffle().to_device(Device::Mps).return_smaller_last_batch() {
                // Iter2::new(&state_tensors, &policy_tensors, args.batch_size).shuffle().return_smaller_last_batch() {

                // optimizer.zero_grad();

                let value_batch = value_tensors.i(idx..min(idx+args.batch_size, num_inputs));

                // let output = self.net.forward_is(&[IValue::from(state_batch)]).unwrap();
                // let (policy, value) = match output {
                //     IValue::Tuple(ivalues) => match &ivalues[..] {
                //         [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
                //         _ => panic!("unexpected output {:?}", ivalues),
                //     },
                //     _ => panic!("unexpected output {:?}", output),
                // };
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


