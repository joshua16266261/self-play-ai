pub mod tictactoe;

use rand::rngs::ThreadRng;

#[derive(Clone, Debug, strum_macros::Display, Default, PartialEq, Eq)]
pub enum Status{
    #[default]
    Ongoing,
    Tied,
    Completed
}

pub trait State: Default + Clone {
    type Encoding: Encoding;
    type Policy: Policy;
    type Player: Eq;

    fn get_current_player(&self) -> Self::Player;
    fn get_next_state(&self, action: &<<Self as crate::game::State>::Policy as Policy>::Action) -> Result<Self, String> where Self: std::marker::Sized;
    fn get_valid_actions(&self) -> Vec<<<Self as crate::game::State>::Policy as Policy>::Action>;
    fn get_value_and_terminated(&self) -> (f32, bool);
    fn encode(&self) -> Self::Encoding;
    fn mask_invalid_actions(&self, policy: Vec<f32>) -> Result<Self::Policy, String>;
}

pub trait Policy: Default {
    type Action: Clone;

    fn get_prob(&self, action: &Self::Action) -> f32;
    fn set_prob(&mut self, action: &Self::Action, prob: f32);
    fn normalize(&mut self);
    fn get_flat_slice(&self) -> &[f32];
    fn sample(&self, rng: &mut ThreadRng, temperature: f32) -> Self::Action;
}

pub trait Encoding {
    fn get_flat_slice(&self) -> &[f32];
    // fn get_original_shape() -> &'static [i64];
}