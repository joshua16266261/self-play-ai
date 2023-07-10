use crate::tictactoe::{State, Action, Policy};
use crate::model::Model;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use tch::nn::VarStore;
// use rand_distr::Dirichlet;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

#[derive(Clone, Copy)]
pub struct Args {
    pub c: f32,
    pub num_searches: u32,
    pub temperature: f32,
    pub num_learn_iters: u32,
    pub num_self_play_iters: u32,
    pub batch_size: i64,
    pub num_epochs: u32
}

struct Node {
    state: State,
    id: usize,
    parent_id: Option<usize>,
    action_taken: Option<Action>,
    prior: f32,
    children_ids: Vec<usize>,
    visit_count: u32,
    value_sum: f32
}

struct Tree {
    args: Args,
    arena: Vec<Node>
}

pub struct MCTS {
    pub args: Args,
    pub model: Model
}

pub struct Learner<'a> {
    pub args: Args,
    pub mcts: MCTS,
    pub var_store: &'a VarStore
}

impl Default for Args {
    fn default() -> Self {
        Args {
            c: f32::sqrt(2.0),
            num_searches: 50,
            temperature: 1.25,
            num_learn_iters: 10,
            num_self_play_iters: 20,
            batch_size: 32,
            num_epochs: 30
        }
    }
}

impl Node {
    fn is_fully_expanded(&self) -> bool {
        !self.children_ids.is_empty()
    }
}

impl Tree {
    fn get_ucb(&self, parent_id: &usize, child_id: &usize) -> f32 {
        let parent_node = self.arena.get(*parent_id).unwrap();
        let child_node = self.arena.get(*child_id).unwrap();
        let q = match child_node.visit_count {
            0 => 0.0,
            _ => 1.0 - (child_node.value_sum / (child_node.visit_count as f32) + 1.0) / 2.0
        };
        q + self.args.c * (parent_node.visit_count as f32).sqrt() / (1.0 + (child_node.visit_count as f32)) * child_node.prior
    }

    fn select(&self, parent_id: &usize) -> &usize {
        let parent_node = self.arena.get(*parent_id).unwrap();
        parent_node.children_ids.iter().max_by(
            |a, b| self.get_ucb(parent_id, a).partial_cmp(&self.get_ucb(parent_id, b)).unwrap()
        ).unwrap()
    }

    fn expand(&mut self, parent_id: usize, policy: Policy) {
        let arena_len = self.arena.len();
        let num_new_nodes = policy.iter().filter(|x| **x > 0.0).count();

        let parent_node = self.arena.get_mut(parent_id).unwrap();
        let mut new_ids: Vec<usize> = (arena_len..(arena_len+num_new_nodes)).collect();
        parent_node.children_ids.append(&mut new_ids);

        let parent_state = parent_node.state.clone();
        let mut new_id = arena_len;
        for row in 0..3 {
            for col in 0..3 {
                let prob = policy[row * 3 + col];
                if prob > 0.0 {
                    let action = Action { row, col };
                    let next_state = parent_state.get_next_state(&action).unwrap();
                    let child_node: Node = Node{
                        state: next_state,
                        id: new_id,
                        parent_id: Some(parent_id),
                        action_taken: Some(action),
                        prior: prob,
                        children_ids: Vec::new(),
                        visit_count: 0,
                        value_sum: 0.0
                    };
                    self.arena.push(child_node);
                    new_id += 1;
                }
            }
        };
    }

    fn backprop(&mut self, node_id: usize, value: f32) {
        let mut node = self.arena.get_mut(node_id).unwrap();

        let mut sign = 1.0;
        node.visit_count += 1;
        node.value_sum += sign * value;
        sign *= -1.0;

        while let Some(parent_id) = node.parent_id {
            node = self.arena.get_mut(parent_id).unwrap();
            node.visit_count += 1;
            node.value_sum += sign * value;
            sign *= -1.0;
        };
    }
}

impl MCTS {
    fn search(&mut self, state: State) -> Policy {
        let root_node_current_player = state.current_player;
        let root_node_id = 0;
        let root_node = Node{
            state,
            id: root_node_id,
            parent_id: None,
            action_taken: None,
            prior: 0.0,
            children_ids: Vec::new(),
            visit_count: 1,
            value_sum: 0.0
        };

        // let (policy, _) = self.model.predict(root_node.state.encode());
        let (policy, _) = self.model.predict(&root_node.state);

        let mut tree = Tree{
            args: self.args,
            arena: vec![root_node]
        };

        // TODO: Add Dirichlet noise
        // let dirichlet = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
        // let mut rng = rand::thread_rng();
        // let samples = dirichlet.sample(&mut rng);

        tree.expand(root_node_id, policy);

        for _ in 0..self.args.num_searches {
            let mut node = tree.arena.get(0).unwrap();
            while node.is_fully_expanded() {
                node = tree.arena.get(*tree.select(&node.id)).unwrap();
            }

            let (mut value, is_terminal) = node.state.get_value_and_terminated();
            if node.state.current_player != root_node_current_player {
                value *= -1.0;
            }

            let node_id = node.id;

            if !is_terminal {
                // let (policy_pred, value_pred) = self.model.predict(node.state.encode());
                let (policy_pred, value_pred) = self.model.predict(&node.state);
                value = value_pred;
                tree.expand(node_id, policy_pred);
            }

            tree.backprop(node_id, value);
        };

        let mut visit_counts: Policy = Default::default();
        let mut total_visit_count = 0.0;
        let children_ids = &tree.arena.get(root_node_id).unwrap().children_ids;
        for child_id in children_ids {
            let child_node = tree.arena.get(*child_id).unwrap();
            let action = child_node.action_taken.clone().unwrap();
            let child_visit_count = child_node.visit_count as f32;

            visit_counts[action.row * 3 + action.col] = child_visit_count;
            total_visit_count += child_visit_count;
        };

        visit_counts.map(|x| x / total_visit_count)
    }
}

impl Learner<'_> {
    fn self_play(&mut self) -> (Vec<[f32; 27]>, Vec<Policy>, Vec<f32>) {
        let mut state_history: Vec<State> = Vec::new();
        let mut policy_history: Vec<Policy> = Vec::new();

        let mut state: State = Default::default();
        let mut rng = rand::thread_rng();
        loop {
            state_history.push(state.clone());
            let action_probs = self.mcts.search(state.clone());
            policy_history.push(action_probs);

            // Higher temperature => squishes probabilities together => encourages more exploration
            let termperature_action_probs: Vec<f32> = action_probs.iter().map(|x| f32::powf(*x, self.args.temperature)).collect();
            let dist = WeightedIndex::new(&termperature_action_probs).unwrap();
            let idx = dist.sample(&mut rng);
            let action = Action{ row: idx / 3, col: idx % 3};
            state = state.get_next_state(&action).unwrap();
            let (value, is_terminal) = state.get_value_and_terminated();

            if is_terminal {
                return (
                    state_history.iter().map(|x| x.encode()).collect(),
                    policy_history,
                    state_history.iter().map(|x| if x.current_player == state.current_player { value } else { -value }).collect() // TODO: Is this correct?
                );
            }
        }
    }

    pub fn learn(&mut self) {
        let multi_pb = MultiProgress::new();
        let pb_style = ProgressStyle::with_template(
            "{prefix:>9} [{elapsed_precise}] {bar:40} {pos:>7}/{len:7} (ETA: {eta_precise}) {msg}",
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

        for i in 0..self.args.num_learn_iters {
            let mut state_memory: Vec<[f32; 27]> = Vec::new();
            let mut policy_memory: Vec<Policy> = Vec::new();
            let mut value_memory: Vec<f32> = Vec::new();

            for _ in 0..self.args.num_self_play_iters {
                let (mut states, mut policies, mut values) = self.self_play();

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
            // self.mcts.model.net.save(format!("../tictactoe_model_{i}.pt")).unwrap();
            self.var_store.save(format!("../tictactoe_model_{i}.pt")).unwrap();

            learn_pb.inc(1);
        }

        learn_pb.finish_and_clear();
        self_play_pb.finish_and_clear();
        train_pb.finish_and_clear();
    }
}
