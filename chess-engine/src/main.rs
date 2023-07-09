mod tictactoe;
mod mcts;
mod model;
use tch::{nn, Tensor, vision::imagenet, IndexOp, IValue, Device};
use tch::display::set_print_options_short;

use crate::model::Model;

fn main() {
    set_print_options_short();

    // TODO: Test Learner

    // Test tictactoe stuff
    // let mut state: tictactoe::State = Default::default();
    // let mut action = tictactoe::Action{row: 0, col: 0};
    // state = state.get_next_state(&action).unwrap();

    // action.row = 2;
    // action.col = 1;
    // state = state.get_next_state(&action).unwrap();
    // println!("{state}");

    // let valid_actions = state.get_valid_actions();
    // let num_valid_actions = valid_actions.len();
    // print!("{num_valid_actions} valid actions:");
    // for action in valid_actions {
    //     print!(" {action}");
    // }
    // println!();

    // action.row = 0;
    // action.col = 1;
    // state = state.get_next_state(&action).unwrap();

    // action.row = 2;
    // action.col = 2;
    // state = state.get_next_state(&action).unwrap();

    // action.row = 0;
    // action.col = 2;
    // state = state.get_next_state(&action).unwrap();

    // let (_, is_terminal) = state.get_value_and_terminated();
    // println!("{state}");
    // println!("Is terminal: {is_terminal}");

    // let encoded_state = state.encode();

    // let mut var_store = nn::VarStore::new(Device::Cpu);
    // let net = tch::TrainableCModule::load("../tictactoe_model.pt", var_store.root()).unwrap();
    // var_store.set_device(Device::Mps);
    
    // let mut model = Model{
    //     net
    // };

    // let (policy, value) = model.predict(encoded_state);
    // println!("{policy:?}");
    // println!("{value}");
}
