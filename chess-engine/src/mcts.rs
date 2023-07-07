use crate::tictactoe::{State, Action, Policy};
use crate::model::Model;

struct Args {
    c: f32
}

struct Node {
    state: State,
    parent_id: Option<usize>,
    action_taken: Action,
    prior: f32,
    children_ids: Vec<usize>,
    visit_count: u32,
    value_sum: f32
}

struct Tree {
    args: Args,
    arena: Vec<Node>
}

struct MCTS {
    args: Args,
    model: Model
}

impl Node {
    fn is_fully_expanded(&self) -> bool {
        !self.children_ids.is_empty()
    }
}

impl Tree {
    fn get_ucb(&self, parent_id: &usize, child_id: &usize) -> Option<f32> {
        let parent_node = self.arena.get(*parent_id)?;
        let child_node = self.arena.get(*child_id)?;
        let q = match child_node.visit_count {
            0 => 0.0,
            _ => 1.0 - (child_node.value_sum / (child_node.visit_count as f32) + 1.0) / 2.0
        };
        Some(q + self.args.c * (parent_node.visit_count as f32).sqrt() / (1.0 + (child_node.visit_count as f32)) * child_node.prior)
    }

    fn select(&self, parent_id: &usize) -> Option<&usize> {
        let parent_node = self.arena.get(*parent_id)?;
        parent_node.children_ids.iter().max_by(
            |a, b| self.get_ucb(parent_id, a).unwrap().partial_cmp(&self.get_ucb(parent_id, b).unwrap()).unwrap()
        )
    }

    fn expand(&mut self, parent_id: &usize, policy: Policy) -> Result<(), String> {
        let arena_len = self.arena.len();
        let num_new_nodes = policy.iter().filter(|x| **x > 0.0).count();

        let parent_node = match self.arena.get_mut(*parent_id) {
            Some(node) => node,
            None => return Err(format!("No node with id {parent_id}"))
        };
        let mut new_ids: Vec<usize> = (arena_len..(arena_len+num_new_nodes)).collect();
        parent_node.children_ids.append(&mut new_ids);

        let parent_state = parent_node.state.clone();

        for row in 0..3 {
            for col in 0..3 {
                let prob = policy[row * 3 + col];
                if prob > 0.0 {
                    let action = Action { row, col };
                    match parent_state.get_next_state(&action) {
                        Ok(child_state) => {
                            let child_node = Node{
                                state: child_state,
                                parent_id: Some(*parent_id),
                                action_taken: action,
                                prior: prob,
                                children_ids: Vec::new(),
                                visit_count: 0,
                                value_sum: 0.0
                            };
                            self.arena.push(child_node);
                        },
                        Err(e) => return Err(e)
                    }
                }
            }
        };
        Ok(())
    }

    fn backprop(&mut self, node_id: &usize, value: f32) -> Result<(), String> {
        let mut node = match self.arena.get_mut(*node_id) {
            Some(node) => node,
            None => return Err(format!("No node with id {node_id}"))
        };

        let mut sign = 1.0;
        node.visit_count += 1;
        node.value_sum += sign * value;
        sign *= -1.0;

        while let Some(parent_id) = node.parent_id {
            node = match self.arena.get_mut(parent_id) {
                Some(node) => node,
                None => return Err(format!("No node with id {node_id}"))
            };
            node.visit_count += 1;
            node.value_sum += sign * value;
            sign *= -1.0;
        };
        
        Ok(())
    }
}

impl MCTS {
    fn search(&self, state: State) -> Policy {

    }
}
