use crate::model::{Model, Net, self};
use crate::mcts::{Mcts, Args, Tree};
use crate::game::{State, Policy};

use core::time;
use std::mem::MaybeUninit;
// use std::sync::mpsc::{channel, Sender, Receiver};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::sleep;
use ringbuf::{SharedRb, Rb, Consumer};
use ndarray::{Array3, Array1, ArrayView3, ArrayView1, stack, Axis, Array};
use tch::{Tensor, Device, kind::Kind, Reduction, nn::{VarStore, Adam, OptimizerConfig, Optimizer}};
use indicatif::ProgressBar;
use rand::{prelude::*, distributions::WeightedIndex};

#[derive(Default)]
pub struct Payload {
    state: Array3<f32>,
    policy: Array1<f32>,
    value: f32
}

pub struct SelfPlayArgs {
    pub c: f32,
    pub num_mcts_searches: u32,
    pub temperature: f32,
    pub num_batched_self_play_games: usize
}

pub struct SelfPlayWorker<T: Net> {
    pub args: SelfPlayArgs,
    pub replay_buffer: Arc<Mutex<SharedRb<Payload, Vec<MaybeUninit<Payload>>>>>,
    pub mcts: Mcts<T>
}

pub struct TrainingArgs {
    pub batch_size: usize,
    pub num_batches_per_iter: u32,
    pub num_train_iters: u32
}

pub struct ModelTrainerWorker<T: Net> {
    pub args: TrainingArgs,
    // rx: Consumer<Payload, Arc<SharedRb<Payload, Vec<MaybeUninit<Payload>>>>>,
    pub replay_buffer: Arc<Mutex<SharedRb<Payload, Vec<MaybeUninit<Payload>>>>>,
    pub net: T,
    pub var_store: VarStore,
    pub optimizer: Optimizer
}

impl Default for SelfPlayArgs {
    fn default() -> Self {
        Self {
            c: 2.0,
            num_mcts_searches: 60,
            temperature: 1.25,
            num_batched_self_play_games: 2
        }
    }
}

impl Default for TrainingArgs {
    fn default() -> Self {
        Self {
            batch_size: 64,
            num_batches_per_iter: 10,
            num_train_iters: 10
        }
    }
}

impl<T: Net> ModelTrainerWorker<T> {
    fn train_batch(&mut self, state_batch: Tensor, policy_batch: Tensor, value_batch: Tensor) -> f64 {
        let (policy, value) = self.net.forward(&state_batch, true);

        let policy_softmax = policy.log_softmax(-1, Kind::Float);
        let policy_nll = policy_softmax * policy_batch;
        let policy_loss = -policy_nll.sum(Kind::Float) / policy.size()[0];

        let value_loss = value.mse_loss(&value_batch, Reduction::Mean);
        let total_loss = policy_loss + value_loss;

        self.optimizer.backward_step(&total_loss);

        total_loss.double_value(&[])
    }

    // TODO: Share Varstore instead of String
    pub fn train_loop(&mut self, checkpoint_path_rwlock: Arc<RwLock<Option<String>>>, checkpoint_dir: &str, pb: &ProgressBar) {
        for i in 0..self.args.num_train_iters {
            for _ in 0..self.args.num_batches_per_iter {
                println!("Waiting for new batch");
                loop {
                    sleep(time::Duration::from_secs(1));
                    let replay_buffer_mutex = self.replay_buffer.lock().unwrap();
                    if replay_buffer_mutex.len() >= self.args.batch_size {
                        break;
                    }
                }

                println!("Training");
                let (state_batch, policy_batch, value_batch) = {
                    let mut replay_buffer_mutex = self.replay_buffer.lock().unwrap();
        
                    let (state_memory,
                        policy_memory,
                        value_memory
                    ): (Vec<_>, Vec<_>, Vec<_>) = replay_buffer_mutex
                        .pop_iter()
                        .take(self.args.batch_size)
                        .fold(
                            (Vec::with_capacity(self.args.batch_size), Vec::with_capacity(self.args.batch_size), Vec::with_capacity(self.args.batch_size)), 
                            |(mut states_acc,
                                mut policies_acc,
                                mut values_acc), payload| {
                                states_acc.push(payload.state);
                                policies_acc.push(payload.policy);
                                values_acc.push(payload.value);
            
                                (states_acc, policies_acc, values_acc)
                            }
                        );
                        
                    let state_memory_views: Vec<ArrayView3<f32>> = state_memory
                        .iter()
                        .map(|x| x.view())
                        .collect();
            
                    let policy_memory_views: Vec<ArrayView1<f32>> = policy_memory
                        .iter()
                        .map(|x| x.view())
                        .collect();
            
                    let state_batch = Tensor::try_from(
                        stack(Axis(0), state_memory_views.as_slice()).unwrap()
                    ).unwrap().to(Device::Mps);
            
                    let policy_batch = Tensor::try_from(
                        stack(Axis(0), policy_memory_views.as_slice()).unwrap()
                    ).unwrap().to(Device::Mps);
            
                    let value_batch = Tensor::try_from(
                        Array::from_vec(value_memory)
                    ).unwrap().view((-1, 1)).to(Device::Mps);

                    (state_batch, policy_batch, value_batch)
                };
                
                let loss = self.train_batch(state_batch, policy_batch, value_batch);
    
                pb.set_message(format!("Loss: {:.3}", loss));
                pb.inc(1);
            }
            println!("Saving checkpoint");
            let latest_checkpoint = format!("{checkpoint_dir}/{i}.safetensors");
            self.var_store.save(latest_checkpoint.clone()).unwrap();
        
            let mut checkpoint_path = checkpoint_path_rwlock.write().unwrap();
            *checkpoint_path = Some(latest_checkpoint);
        }
    }
}

impl<T: Net> SelfPlayWorker<T> {
    fn self_play(&self) -> (Vec<Array3<f32>>, Vec<Array1<f32>>, Vec<f32>) {
        let mut state_history: Vec<Array3<f32>> = Vec::new();
        let mut policy_history: Vec<Array1<f32>> = Vec::new();
        let mut value_history: Vec<f32> = Vec::new();

        let mut trees = vec![Tree::<T::State>::default(); self.args.num_batched_self_play_games];
        let mut trees_vec: Vec<_> = trees.iter_mut().collect();

        let mut rng = rand::thread_rng();

        while !trees_vec.is_empty() {
            let mut search_results = self.mcts.search(&mut trees_vec);

            for i in (0..trees_vec.len()).rev() {
                let tree = trees_vec.get_mut(i).unwrap();
                let root_state = tree.arena.get(0).unwrap().state.clone();

                let (action_probs, child_id_to_probs) = search_results.pop().unwrap();

                // Higher temperature => squishes probabilities together => encourages more exploration
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
                            .iter()
                            .map(|x| x.get_encoding())
                            .collect()
                    );

                    policy_history.append(
                        &mut tree.policy_history
                            .iter_mut()
                            .map(|policy| policy.get_flat_ndarray())
                            .collect()
                    );

                    value_history.append(
                        &mut tree.state_history
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

                    trees_vec.remove(i);
                } else {
                    // Next search starts from new sampled state
                    tree.node_id_to_expand = None;
                    // tree.arena = vec![Node::create_root_node(state.clone())];
                    tree.use_subtree(selected_id);
                }
            }
        }
        
        (state_history, policy_history, value_history)
    }

    pub fn self_play_loop(&mut self, checkpoint_path_rwlock: Arc<RwLock<Option<String>>>) {
        loop {
            println!("Getting latest model");

            let mut var_store = VarStore::new(Device::Cpu);
            let net = T::new(
                &var_store.root(),
                Default::default()
            );
            if let Some(path) = checkpoint_path_rwlock.read().unwrap().clone() {
                var_store.load(path).unwrap();
            }
            var_store.set_device(Device::Mps);
            self.mcts.model.net = net;

            println!("Self-play");
            let (
                state_history,
                policy_history,
                value_history
            ) = self.self_play();

            println!("Pushing to replay buffer");
            let payload_iter = state_history.iter().zip(policy_history.iter().zip(value_history.iter()))
                .map(|(state, (policy, value))| 
                    Payload {
                        state: state.clone(),
                        policy: policy.clone(),
                        value: *value
                    }
                );

            let mut replay_buffer_mutex = self.replay_buffer.lock().unwrap();
            replay_buffer_mutex.push_iter_overwrite(payload_iter);
        }
    }
}

