use crate::game::chess;
use tch::{Tensor, nn::{SequentialT, ModuleT}};
use tch::nn;

pub struct Net {
    torso: SequentialT,
    policy_head: SequentialT,
    value_head: SequentialT
}

impl Net {
    pub fn new(vs: &nn::Path, num_resnet_blocks: u32, num_hidden: i64) -> Net {
        let conv_config = nn::ConvConfig { padding: 0, ..Default::default() };
        Net {
            torso: super::new_resnet(vs, num_resnet_blocks, 19, num_hidden),
            policy_head: nn::seq_t()
                .add(nn::conv2d(vs, num_hidden, 256, 1, conv_config))
                .add_fn(|x| x.relu())
                .add(nn::conv2d(vs, 256, 73, 1, conv_config))
                .add_fn(|x| x.flat_view()),
            value_head: nn::seq_t()
                .add(nn::conv2d(vs, num_hidden, 1, 1, conv_config))
                .add_fn(|x| x.relu())
                .add_fn(|x| x.flat_view())
                .add(nn::linear(vs, 8 * 8, 256, Default::default()))
                .add_fn(|x| x.relu())
                .add(nn::linear(vs, 256, 1, Default::default()))
                .add_fn(|x| x.tanh())
        }
    }
}

impl super::Net for Net {
    type State = chess::State;

    fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let torso_output = self.torso.forward_t(x, train);
        let policy = self.policy_head.forward_t(&torso_output, train);
        let value = self.value_head.forward_t(&torso_output, train);

        (policy, value)
    }
}
