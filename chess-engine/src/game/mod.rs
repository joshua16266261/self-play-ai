pub mod tictactoe;

use rand::rngs::ThreadRng;

#[derive(Copy, Clone, Debug, strum_macros::Display, Default, PartialEq, Eq)]
pub enum Status{
    #[default]
    Ongoing,
    Tied,
    Completed
}

pub trait Player: Eq + Copy {
    fn get_opposite(&self) -> Self;
}

pub trait State: Default + Clone {
    type Encoding: Encoding;
    type Policy: Policy;
    type Player: Player;

    fn get_current_player(&self) -> Self::Player;
    fn get_next_state(&self, action: &<<Self as crate::game::State>::Policy as Policy>::Action) ->
        Result<Self, String> where Self: std::marker::Sized;
    fn get_valid_actions(&self) -> Vec<<<Self as crate::game::State>::Policy as Policy>::Action>;
    fn get_status(&self) -> Status;
    fn get_value_and_terminated(&self) -> (f32, bool);
    fn encode(&self) -> Self::Encoding;
    fn mask_invalid_actions(&self, policy: Vec<f32>) -> Result<Self::Policy, String>;
}

pub trait Policy: Default + Copy {
    type Action: Clone;

    fn get_prob(&self, action: &Self::Action) -> f32;
    fn set_prob(&mut self, action: &Self::Action, prob: f32);
    fn get_normalized(&mut self) -> Self;
    fn get_flat_slice(&self) -> &[f32];
    fn sample(&self, rng: &mut ThreadRng, temperature: f32) -> Self::Action;
    fn get_best_action(&self) -> Self::Action;
}

pub trait Encoding {
    fn get_flat_slice(&self) -> &[f32];
}
