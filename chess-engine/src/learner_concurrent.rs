use crate::model::Net;
use crate::mcts::{Mcts, Tree};
use crate::game::{State, Policy};

use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, RwLock, Condvar};
use ringbuf::{SharedRb, Rb};
use ndarray::{Array3, Array1, ArrayView3, ArrayView1, stack, Axis, Array};
use tch::{Tensor, Device, kind::Kind, Reduction, nn::{VarStore, Optimizer}};
use indicatif::ProgressBar;
use rand::{prelude::*, distributions::WeightedIndex};

#[derive(Default)]
pub struct Payload {
    state: Array3<f32>,
    policy: Array1<f32>,
    value: f32
}

#[derive(Clone)]
pub struct SelfPlayArgs {
    pub c: f32,
    pub num_mcts_searches: u32,
    pub temperature: f32,
    pub num_batched_self_play_games: usize
}

pub type ReplayBuffer = Arc<(Mutex<SharedRb<Payload, Vec<MaybeUninit<Payload>>>>, Condvar)>;

pub struct SelfPlayWorker<T: Net> {
    pub args: SelfPlayArgs,
    pub replay_buffer: ReplayBuffer,
    pub mcts: Mcts<T>
}

pub struct TrainingArgs {
    pub batch_size: usize,
    pub num_batches_per_iter: u32,
    pub num_train_iters: u32
}

pub struct ModelTrainerWorker<T: Net> {
    pub args: TrainingArgs,
    pub replay_buffer: ReplayBuffer,
    pub net: T,
    pub var_store: VarStore,
    pub optimizer: Optimizer
}

impl Default for SelfPlayArgs {
    fn default() -> Self {
        Self {
            c: 2.0,
            num_mcts_searches: 600,
            temperature: 1.25,
            num_batched_self_play_games: 100
        }
    }
}

impl Default for TrainingArgs {
    fn default() -> Self {
        Self {
            batch_size: 128,
            num_batches_per_iter: 20,
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

    pub fn train_loop(
        &mut self,
        varstore_rwlock: Arc<RwLock<VarStore>>,
        in_training_rwlock: Arc<RwLock<bool>>,
        checkpoint_dir: &str,
        training_pb: &ProgressBar,
        replay_buffer_pb: Arc<Mutex<ProgressBar>>
    ) {

        for i in 0..self.args.num_train_iters {
            for _ in 0..self.args.num_batches_per_iter {
                let (state_batch, policy_batch, value_batch) = {
                    let (replay_buffer_mutex, cvar) = &*self.replay_buffer;
                    let mut replay_buffer = cvar.wait_while(
                        replay_buffer_mutex.lock().unwrap(), 
                        |replay_buffer| replay_buffer.len() < self.args.batch_size
                    ).unwrap();
        
                    let (state_memory,
                        policy_memory,
                        value_memory
                    ): (Vec<_>, Vec<_>, Vec<_>) = replay_buffer
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

                    replay_buffer_pb.lock().unwrap().set_position(replay_buffer.len() as u64);
                        
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
                training_pb.set_message(format!("Loss: {:.3}", loss));
            }

            let latest_checkpoint = format!("{checkpoint_dir}/{i}.safetensors");
            self.var_store.save(latest_checkpoint.clone()).unwrap();
        
            let mut var_store = varstore_rwlock.write().unwrap();
            var_store.copy(&self.var_store).unwrap();

            training_pb.inc(1);
        }

        *in_training_rwlock.write().unwrap() = false;
    }
}

impl<T: Net> SelfPlayWorker<T> {
    fn self_play(&self, worker_pb: &ProgressBar) -> (Vec<Array3<f32>>, Vec<Array1<f32>>, Vec<f32>) {
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
                    tree.use_subtree(selected_id);
                }
            }

            worker_pb.inc(1);
        }
        
        (state_history, policy_history, value_history)
    }

    pub fn self_play_loop(
        &mut self,
        checkpoint_path_rwlock: Arc<RwLock<VarStore>>,
        in_training_rwlock: Arc<RwLock<bool>>,
        worker_pb: &ProgressBar,
        replay_buffer_pb: Arc<Mutex<ProgressBar>>
    ) {

        let mut rng = rand::thread_rng();

        loop {
            if !*in_training_rwlock.read().unwrap() {
                worker_pb.finish_and_clear();
                break;
            }

            let mut var_store = VarStore::new(Device::Cpu);
            var_store.copy(&checkpoint_path_rwlock.read().unwrap()).unwrap();
            let net = T::new(&var_store.root(),Default::default());
            var_store.set_device(Device::Mps);
            self.mcts.model.net = net;

            let (
                state_history,
                policy_history,
                value_history
            ) = self.self_play(worker_pb);

            let payload_iter = state_history.iter().zip(policy_history.iter().zip(value_history.iter()))
                .map(|(state, (policy, value))| 
                    Payload {
                        state: state.clone(),
                        policy: policy.clone(),
                        value: *value
                    }
                );

            let payload_shuffled = payload_iter.choose_multiple(&mut rng, (state_history.len() as f32 * 0.3) as usize);

            let (replay_buffer, cvar) = &*self.replay_buffer;
            let mut replay_buffer_mutex = replay_buffer.lock().unwrap();
            replay_buffer_mutex.push_iter_overwrite(payload_shuffled.into_iter());
            cvar.notify_one();

            replay_buffer_pb.lock().unwrap().set_position(replay_buffer_mutex.len() as u64);
        }
    }
}

