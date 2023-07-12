use crate::game::tictactoe;
use tch::{Tensor, nn::{SequentialT, ModuleT}};
use tch::nn;

// TODO: Separate Model and TicTacToeNet

pub struct Net {
    torso: SequentialT,
    policy_head: SequentialT,
    value_head: SequentialT
}

impl Net {
    pub fn new(vs: &nn::Path, num_resnet_blocks: u32, num_hidden: i64) -> Net {
        let conv_config = nn::ConvConfig { padding: 1, ..Default::default() };
        Net {
            torso: nn::seq_t()
                .add_fn(|x| x.view((-1, 3, 3, 3)))
                .add(super::new_resnet(vs, num_resnet_blocks, 3, num_hidden)),
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
}

impl super::Net for Net {
    type State = tictactoe::State;

    fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let torso_output = self.torso.forward_t(x, train);
        let policy = self.policy_head.forward_t(&torso_output, train);
        let value = self.value_head.forward_t(&torso_output, train);

        (policy, value)
    }
}
