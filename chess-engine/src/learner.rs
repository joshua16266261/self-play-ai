use crate::game::{State, Policy};
use crate::model::Net;
use crate::mcts::{Args, Mcts, Node, Tree};

use ndarray::{Array3, Array1, Array, stack, Axis, ArrayView3, ArrayView1};
use rayon::prelude::*;
use rayon::iter::repeatn;
use tch::{Tensor, Device, nn::VarStore};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::fs::create_dir;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::time::Instant;

pub struct Learner<'a, T: Net> {
    pub args: Args,
    pub mcts: Mcts<T>,
    pub var_store: &'a mut VarStore
}

impl<T: Net> Learner<'_, T> {
    fn self_play(&self) -> (Vec<Array3<f32>>, Vec<Array1<f32>>, Vec<f32>) {
        let mut state_history: Vec<Array3<f32>> = Vec::new();
        let mut policy_history: Vec<Array1<f32>> = Vec::new();
        let mut value_history: Vec<f32> = Vec::new();

        let mut trees: Vec<_> = repeatn(Tree::<T::State>::default(), self.args.num_parallel_self_play_games).collect();
        let mut trees_vec: Vec<_> = trees.par_iter_mut().collect();

        let mut rng = rand::thread_rng();

        while !trees_vec.is_empty() {
            let now = Instant::now();
            let mut search_results = self.mcts.search(&mut trees_vec);

            for i in (0..trees_vec.len()).rev() {
                let tree = trees_vec.get_mut(i).unwrap();
                let root_state = tree.arena.get(0).unwrap().state.clone();

                let (action_probs, child_id_to_probs) = search_results.pop().unwrap();

                // Higher temperature => squishes probabilities together => encourages more exploration
                // let action = action_probs.sample(&mut rng, self.args.temperature);
                // let state = root_state.get_next_state(&action).unwrap();
                let temperature_action_probs = child_id_to_probs
                    .iter()
                    .map(|(_, prob)| prob.powf(self.args.temperature));
                let dist = WeightedIndex::new(temperature_action_probs).unwrap();
                let idx = dist.sample(&mut rng);
                let selected_id = child_id_to_probs[idx].0;
                let state = &tree.arena[selected_id].state;

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

                    trees_vec.remove(i);
                } else {
                    // Next search starts from new sampled state
                    tree.node_id_to_expand = None;
                    tree.use_subtree(selected_id);
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
            let (state_memory,
                policy_memory,
                value_memory): (Vec<_>, Vec<_>, Vec<_>) = repeatn({
                    let (
                        states,
                        policies,
                        values
                    ) = self.self_play();

                    self_play_pb.inc(1);

                    (states, policies, values)
                },self.args.num_self_play_iters as usize)
                .reduce(
                || (Vec::new(), Vec::new(), Vec::new()),
                |
                    (
                        mut states_acc,
                        mut policies_acc,
                        mut values_acc
                    ),
                    (
                        mut states,
                        mut policies,
                        mut values
                    )
                    | {
                        states_acc.append(&mut states);
                        policies_acc.append(&mut policies);
                        values_acc.append(&mut values);

                        (states_acc, policies_acc, values_acc)
                    }
                );

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
    }
}
