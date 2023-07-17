use crate::game::{State, Policy};
use crate::model::Net;
use crate::mcts::{Args, Mcts, Node, Tree};

use ndarray::{Array3, Array1, Array, stack, Axis, ArrayView3, ArrayView1};
use rayon::prelude::*;
use rayon::iter::repeatn;
use tch::{Tensor, Device, nn::VarStore};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::fs::create_dir;

pub struct Learner<'a, T: Net> {
    pub args: Args,
    pub mcts: Mcts<T>,
    pub var_store: &'a mut VarStore
}

impl<T: Net> Learner<'_, T> {
    #[allow(clippy::type_complexity)]
    fn self_play(&self) -> (
        Vec<Array3<f32>>,
        Vec<Array1<f32>>,
        Vec<f32>
    ) {

        let mut state_history: Vec<Array3<f32>> = Vec::new();
        let mut policy_history: Vec<Array1<f32>> = Vec::new();
        let mut value_history: Vec<f32> = Vec::new();

        let mut trees: Vec<Tree<T::State>> = repeatn(Tree::<T::State>::default(), self.args.num_parallel_self_play_games).collect();

        let mut rng = rand::thread_rng();

        while !trees.is_empty() {
            let mut all_action_probs = self.mcts.search(&mut trees);
            // let all_action_probs = Policy::to_ndarray(self.mcts.search(&mut trees));

            // TODO: Might be able to parallelize
            // Handle removal with filter function
            for i in (0..trees.len()).rev() {
                // TODO: Use map_with to map terminal trees to their state, policy, and value histories
                // Non-terminal trees will be mapped to zero-capacity vectors
                // and the trees themselves will be a sent to a receiver to be collected and used for the next iteration
                // The result of map_with will be flattened and collected

                let tree = trees.get_mut(i).unwrap();
                let root_state = tree.arena.get(0).unwrap().state.clone();

                // let action_probs = *all_action_probs.get(i).unwrap();
                let action_probs = all_action_probs.pop().unwrap();

                // Higher temperature => squishes probabilities together => encourages more exploration
                let action = action_probs.sample(&mut rng, self.args.temperature);
                let state = root_state.get_next_state(&action).unwrap();

                tree.state_history.push(root_state);
                tree.policy_history.push(action_probs);

                let (value, is_terminal) = state.get_value_and_terminated();
                if is_terminal {
                    state_history.append(
                        &mut tree.state_history
                            .par_iter()
                            .map(|x| x.get_encoding())
                            .collect()
                    );

                    policy_history.append(
                        &mut tree.policy_history
                            .par_iter_mut()
                            .map(|policy| policy.get_flat_ndarray())
                            .collect()
                    );

                    value_history.append(
                        &mut tree.state_history
                            .par_iter()
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

                    trees.remove(i);
                } else {
                    // Next search starts from new sampled state
                    tree.arena = vec![Node::create_root_node(state)];
                    tree.node_id_to_expand = None;
                }
            }
        }

        (state_history, policy_history, value_history)
    }

    pub fn learn(&self, checkpoint_dir: &str) {
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
            let mut state_memory: Vec<Array3<f32>> = Vec::new();
            let mut policy_memory: Vec<Array1<f32>> = Vec::new();
            let mut value_memory: Vec<f32> = Vec::new();

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

            let state_memory_views: Vec<ArrayView3<f32>> = state_memory
                .iter()
                .map(|x| x.view())
                .collect();

            let policy_memory_views: Vec<ArrayView1<f32>> = policy_memory
                .iter()
                .map(|x| x.view())
                .collect();

            let state_memory = Tensor::try_from(
                stack(Axis(0), state_memory_views.as_slice()).unwrap()
            ).unwrap().to(Device::Mps);

            let policy_memory = Tensor::try_from(
                stack(Axis(0), policy_memory_views.as_slice()).unwrap()
            ).unwrap().to(Device::Mps);

            let value_memory = Tensor::try_from(
                Array::from_vec(value_memory)
            ).unwrap().view((-1, 1)).to(Device::Mps);

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
