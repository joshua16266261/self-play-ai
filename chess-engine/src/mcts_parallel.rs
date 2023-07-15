use crate::game::{State, Policy};
use crate::model::{Net, Model};

// use rand_distr::Dirichlet;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct Args {
    pub c: f32,
    pub num_searches: u32,
    pub temperature: f32,
    pub num_learn_iters: u32,
    pub num_self_play_iters: u32,
    pub num_parallel_self_play_games: usize,
    pub batch_size: i64,
    pub num_epochs: u32
}

#[derive(Clone)]
pub struct Node<T: State> {
    pub state: T,
    id: usize,
    parent_id: Option<usize>,
    action_taken: Option<<<T as State>::Policy as Policy>::Action>,
    prior: Option<f32>,
    children_ids: Vec<usize>,
    visit_count: u32,
    value_sum: f32
}

#[derive(Clone)]
pub struct Tree<T: State> {
    pub args: Args,
    pub arena: Vec<Node<T>>,
    pub node_id_to_expand: Option<usize>, 
    pub state_history: Vec<T>,
    pub policy_history: Vec<T::Policy>
}

pub struct Mcts<T: Net> {
    pub args: Args,
    pub model: Model<T>
}

impl Default for Args {
    fn default() -> Self {
        Args {
            c: 2.0,
            num_searches: 1000,
            temperature: 1.25,
            num_learn_iters: 3,
            num_self_play_iters: 500,
            num_parallel_self_play_games: 1,
            batch_size: 64,
            num_epochs: 4
        }
    }
}

impl<T: State> Node<T> {
    fn is_fully_expanded(&self) -> bool {
        !self.children_ids.is_empty()
    }
}

impl<T: State> Default for Node<T> {
    fn default() -> Self {
        Self {
            state: T::default(),
            id: 0,
            parent_id: None,
            action_taken: None,
            prior: None,
            children_ids: Vec::new(),
            visit_count: 1,
            value_sum: 0.0
        }
    }
}

impl<T: State> Default for Tree<T> {
    fn default() -> Self { 
        Self {
            args: Args::default(),
            arena: vec![Node::default()],
            node_id_to_expand: None,
            state_history: Vec::new(),
            policy_history: Vec::new()
        }
    }
}

impl<T: State> Node<T> {
    pub fn create_root_node(state: T) -> Self {
        Self { state, ..Default::default() }
    }
}

impl<T: State> Tree<T> {
    pub fn with_root_state(state: T) -> Self {
        let root_node = Node::create_root_node(state);
        Self { arena: vec![root_node], ..Default::default() }
    }

    fn get_ucb(&self, parent_id: usize, child_id: usize) -> f32 {
        let parent_node = self.arena.get(parent_id).unwrap();
        let child_node = self.arena.get(child_id).unwrap();
        let q = match child_node.visit_count {
            0 => 0.0,
            // 0 => 0.5, // TODO: Is this correct?
            // 0 => 0.9,
            _ => (-child_node.value_sum / (child_node.visit_count as f32) + 1.0) / 2.0
        };
        q + self.args.c * child_node.prior.unwrap() * (parent_node.visit_count as f32).sqrt() / (1.0 + (child_node.visit_count as f32))
    }

    fn select(&self, parent_id: usize) -> usize {
        // TODO: Parallelize
        let parent_node = self.arena.get(parent_id).unwrap();
        let node_comparison = |a: &&usize, b: &&usize|
            self.get_ucb(parent_id, **a)
            .partial_cmp(
                &self.get_ucb(parent_id, **b)
            )
            .unwrap();
        *parent_node.children_ids
            .iter()
            .max_by(node_comparison)
            .unwrap()
    }

    fn expand(&mut self, parent_id: usize, policy: T::Policy) {
        let arena_len = self.arena.len();
        let parent_node = self.arena.get_mut(parent_id).unwrap();
        let num_new_nodes = parent_node.state.get_valid_actions().len();

        let mut new_ids: Vec<usize> = (arena_len..(arena_len+num_new_nodes)).collect();
        parent_node.children_ids.append(&mut new_ids);

        let parent_state: T = parent_node.state.clone();
        let mut new_id = arena_len;

        for action in parent_state.get_valid_actions() {
            let prob = policy.get_prob(&action);
            let next_state = parent_state.get_next_state(&action).unwrap();
            let child_node = Node {
                state: next_state,
                id: new_id,
                parent_id: Some(parent_id),
                action_taken: Some(action),
                prior: Some(prob),
                children_ids: Vec::new(),
                visit_count: 0,
                value_sum: 0.0
            };
            self.arena.push(child_node);
            new_id += 1;
        }
    }

    fn backprop(&mut self, node_id: usize, value: f32) {
        let mut node = self.arena.get_mut(node_id).unwrap();

        let mut sign = 1.0;
        node.visit_count += 1;
        node.value_sum += sign * value;
        sign *= -1.0;

        // TODO: If we include a depth field to the Node struct,
        // then this might be parallelizable
        while let Some(parent_id) = node.parent_id {
            node = self.arena.get_mut(parent_id).unwrap();
            node.visit_count += 1;
            node.value_sum += sign * value;
            sign *= -1.0;
        };
    }
}

impl<T: Net> Mcts<T> {
    pub fn search(&self, trees: &mut Vec<Tree<T::State>>) -> Vec<<<T as Net>::State as State>::Policy> {
        let states = trees
            .par_iter_mut()
            .map(|x| &x.arena.get(0).unwrap().state)
            .collect();

        // TODO: After changing Model to be consistent, change states to be a vector of encodings
        let (policies, _) = self.model.predict(&states);

        // TODO: Add Dirichlet noise
        // let dirichlet = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
        // let mut rng = rand::thread_rng();
        // let samples = dirichlet.sample(&mut rng);

        trees.par_iter_mut().zip(policies)
            .for_each(|(tree, policy)|
                tree.expand(0, policy)
            );

        for _ in 0..self.args.num_searches {
            // let mut trees_to_expand: Vec<&mut Tree<T::State>> = trees
            //     .par_iter_mut()
            //     .update(|tree| {
            //         let mut node = tree.arena.get(0).unwrap();

            //         while node.is_fully_expanded() {
            //             node = tree.arena.get(tree.select(node.id)).unwrap();
            //         }

            //         let (value, is_terminal) = node.state.get_value_and_terminated();

            //         if is_terminal {
            //             tree.backprop(node.id, value);
            //             tree.node_id_to_expand = None;
            //         } else {
            //             tree.node_id_to_expand = Some(node.id);
            //         }
            //     })
            //     .filter(|tree| tree.node_id_to_expand.is_some())
            //     .collect();

            for tree in trees.iter_mut() {
                let mut node = tree.arena.get(0).unwrap();

                while node.is_fully_expanded() {
                    node = tree.arena.get(tree.select(node.id)).unwrap();
                }

                let (value, is_terminal) = node.state.get_value_and_terminated();

                if is_terminal {
                    tree.backprop(node.id, value);
                    tree.node_id_to_expand = None;
                } else {
                    tree.node_id_to_expand = Some(node.id);
                }
            }

            let mut trees_to_expand: Vec<&mut Tree<T::State>> = trees
                .iter_mut()
                // .filter_map(|tree| {
                //     if tree.node_id_to_expand.is_some() {
                //         Some(tree)
                //     } else {
                //         None
                //     }
                // })
                .filter(|tree| tree.node_id_to_expand.is_some())
                .collect();
            
            if !trees_to_expand.is_empty() {
                let states = trees_to_expand
                    .par_iter_mut()
                    .map(|tree|
                        &tree.arena.get(tree.node_id_to_expand.unwrap()).unwrap().state
                    )
                    .collect();

                let (policies, values) = self.model.predict(&states);

                (trees_to_expand, policies, values).into_par_iter()
                    .for_each(
                        |(tree, policy, value)| {
                            let node_id = tree.node_id_to_expand.unwrap();
                            tree.expand(node_id, policy);
                            tree.backprop(node_id, value);
                        }
                    );
            }
        };

        trees
            .par_iter_mut()
            .map(|tree| {
                let mut visit_counts = <<T as crate::model::Net>::State as State>::Policy::default();
                let children_ids = &tree.arena.get(0).unwrap().children_ids;
                for child_id in children_ids {
                    let child_node = tree.arena.get(*child_id).unwrap();
                    let action = child_node.action_taken.clone().unwrap();
                    let child_visit_count = child_node.visit_count as f32;

                    visit_counts.set_prob(&action, child_visit_count);
                }

                visit_counts.normalize();
                visit_counts
            })
            .collect()
    }
}
