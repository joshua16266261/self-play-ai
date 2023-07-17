pub mod tictactoe;
pub mod chess;

use rand::rngs::ThreadRng;
use ndarray::{Array1, Array3, ArrayView1};
use std::fmt;

#[derive(Copy, Clone, Debug, strum_macros::Display, Default, PartialEq, Eq)]
pub enum Status{
    #[default]
    Ongoing,
    Tied,
    Won
}

pub trait Player: Eq + Copy {
    fn get_opposite(&self) -> Self;
}

pub trait State: Default + Clone + Send + Sync {
    type Policy: Policy;
    type Player: Player;

    fn get_current_player(&self) -> Self::Player;
    fn get_next_state(&self, action: &<<Self as crate::game::State>::Policy as Policy>::Action) -> Result<Self, String>;
    fn get_valid_actions(&self) -> Vec<<<Self as crate::game::State>::Policy as Policy>::Action>;
    fn get_status(&self) -> Status;
    fn get_value_and_terminated(&self) -> (f32, bool);
    fn get_encoding(&self) -> Array3<f32>;
    fn mask_invalid_actions(&self, policy: ArrayView1<f32>) -> Result<Self::Policy, String>;
}

pub trait Policy: Default + Send + Clone {
    type Action: Clone + Send + fmt::Debug;

    fn get_prob(&self, action: &Self::Action) -> f32;
    fn set_prob(&mut self, action: &Self::Action, prob: f32);
    fn normalize(&mut self);
    fn get_flat_ndarray(&self) -> Array1<f32>;
    fn sample(&self, rng: &mut ThreadRng, temperature: f32) -> Self::Action;
    fn get_best_action(&self) -> Self::Action;
}
