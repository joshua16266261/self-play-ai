use crate::tictactoe::Policy;
use tch::{Tensor, TrainableCModule, IValue, Device};

pub struct Model {
    pub net: TrainableCModule
}

impl Model {
    pub fn predict(&mut self, state: [f32; 27]) -> (Policy, f32) {
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

        // let policy_cpu = policy.to(Device::Cpu);
        // let value_cpu = value.to(Device::Cpu);
        // println!("{policy_cpu}");
        // println!("{value_cpu}");

        let policy_vec = Vec::<f32>::try_from(policy.view(-1)).unwrap();
        let value_float = value.double_value(&[0]) as f32;

        let mut return_policy = [0f32; 9];
        for (i, prob) in return_policy.iter_mut().enumerate() {
            *prob = *policy_vec.get(i).unwrap();
        }

        (return_policy, value_float)
    }
}