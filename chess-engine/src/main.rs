mod tictactoe;
mod mcts;
mod model;
use tch::{Tensor};
use tch::display::set_print_options_short;

fn main() {
    set_print_options_short();

    // Test tensor stuff
    let array = [0, 1, 2, 3, 4, 5];
    let t = Tensor::from_slice(&array).view((2, 3));
    let s = t.f_add_scalar(2.5).unwrap();
    println!("{s}");

    // Test tictactoe stuff
    let mut state: tictactoe::State = Default::default();
    let mut action = tictactoe::Action{row: 0, col: 0};
    state = state.get_next_state(&action).unwrap();

    action.row = 2;
    action.col = 1;
    state = state.get_next_state(&action).unwrap();
    println!("{state}");

    let valid_actions = state.get_valid_actions();
    let num_valid_actions = valid_actions.len();
    print!("{num_valid_actions} valid actions:");
    for action in valid_actions {
        print!(" {action}");
    }
    println!();

    action.row = 0;
    action.col = 1;
    state = state.get_next_state(&action).unwrap();

    action.row = 2;
    action.col = 2;
    state = state.get_next_state(&action).unwrap();

    action.row = 0;
    action.col = 2;
    state = state.get_next_state(&action).unwrap();

    let (_, is_terminal) = state.get_value_and_terminated();
    println!("{state}");
    println!("Is terminal: {is_terminal}");

    let encoded_state = Tensor::from_slice(&state.encode()).view((3, 3, 3));
    println!("{encoded_state}");
}
