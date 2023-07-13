use crate::game::{State, Policy};
use crate::model::Net;
use crate::mcts::{Args, Mcts};

use tch::nn::VarStore;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::fs::create_dir;

pub struct Learner<'a, T: Net> {
    pub args: Args,
    pub mcts: Mcts<T>,
    pub var_store: &'a mut VarStore
}

impl<T: Net> Learner<'_, T> {
    #[allow(clippy::type_complexity)]
    fn self_play(&mut self) -> (
        Vec<<<T as crate::model::Net>::State as State>::Encoding>,
        Vec<<<T as crate::model::Net>::State as State>::Policy>,
        Vec<f32>
    ) {
        let mut state_history: Vec<T::State> = Vec::new();
        let mut policy_history: Vec<<<T as crate::model::Net>::State as State>::Policy> = Vec::new();

        let mut state = T::State::default();
        let mut rng = rand::thread_rng();

        loop {
            state_history.push(state.clone());
            let action_probs = self.mcts.search(state.clone());
            policy_history.push(action_probs);

            // Higher temperature => squishes probabilities together => encourages more exploration
            let action = action_probs.sample(&mut rng, self.args.temperature);
            state = state.get_next_state(&action).unwrap();
            let (value, is_terminal) = state.get_value_and_terminated();

            if is_terminal {
                return (
                    state_history.iter().map(|x| x.encode()).collect(),
                    policy_history,
                    state_history
                        .iter()
                        .map(
                            |x|
                                if x.get_current_player() == state.get_current_player() {
                                    value
                                } else {
                                    -value
                                }
                        )
                        .collect()
                );
            }
        }
    }

    pub fn learn(&mut self, checkpoint_dir: &str) {
        let _ = create_dir(checkpoint_dir);

        // Progress bars
        let multi_pb = MultiProgress::new();
        let pb_style = ProgressStyle::with_template(
            "{prefix:>9} [{elapsed_precise}] {bar:40} {pos:>7}/{len:7} (ETA: {eta_precise}) {msg}"
        )
            .unwrap()
            .progress_chars("##-");

        let learn_pb = multi_pb.add(ProgressBar::new(self.args.num_learn_iters as u64));
        learn_pb.set_style(pb_style.clone());
        learn_pb.set_prefix("Learning");

        let self_play_pb = multi_pb.insert_after(&learn_pb, ProgressBar::new(self.args.num_self_play_iters as u64));
        self_play_pb.set_style(pb_style.clone());
        self_play_pb.set_prefix("Self-play");

        let train_pb = multi_pb.insert_after(&self_play_pb, ProgressBar::new(self.args.num_epochs as u64));
        train_pb.set_style(pb_style);
        train_pb.set_prefix("Training");

        train_pb.finish_and_clear();

        // Actual logic
        for i in 0..self.args.num_learn_iters {
            let mut state_memory: Vec<<<T as crate::model::Net>::State as State>::Encoding> = Vec::new();
            let mut policy_memory: Vec<<<T as crate::model::Net>::State as State>::Policy> = Vec::new();
            let mut value_memory: Vec<f32> = Vec::new();

            // TODO: Parallelize
            for _ in 0..self.args.num_self_play_iters {
                let (
                    mut states,
                    mut policies,
                    mut values
                ) = self.self_play();

                state_memory.append(&mut states);
                policy_memory.append(&mut policies);
                value_memory.append(&mut values);

                self_play_pb.inc(1);
            }

            self_play_pb.reset();

            self.mcts.model.train(
                state_memory, 
                policy_memory, 
                value_memory, 
                self.var_store, 
                self.args, 
                &train_pb
            );
            self.var_store.save(format!("{checkpoint_dir}/{i}.safetensors")).unwrap();

            learn_pb.inc(1);
        }

        // learn_pb.finish_and_clear();
        // self_play_pb.finish_and_clear();
        // train_pb.finish_and_clear();
    }
}
