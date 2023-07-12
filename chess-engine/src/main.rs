mod mcts;
mod model;
mod game;

use model::{Model, TicTacToeNet};
use mcts::{Args, Learner, Mcts};
use tch::{nn, Device};
use tch::display::set_print_options_short;
use game::{State, Policy};
use game::tictactoe;

use crate::game::Player;

// TODO: Test new refactored code

fn train() {
    let mut var_store = nn::VarStore::new(Device::Cpu);
    let net = TicTacToeNet::new(&var_store.root(), 4, 64);
    var_store.set_device(Device::Mps);

    let args = Args::default();
    let model = Model{ args, net };
    let mcts = Mcts{ args, model };

    let mut learner = Learner{
        args,
        mcts,
        var_store: &mut var_store
    };

    learner.learn::<tictactoe::State>();
}

fn play() {
    let mut var_store = nn::VarStore::new(Device::Cpu);
    let net = TicTacToeNet::new(&var_store.root(), 4, 64);
    var_store.load("../tictactoe_model_2.safetensors").unwrap();
    var_store.set_device(Device::Mps);

    let args = Args::default();
    let model = Model{ args, net };
    let mut mcts = Mcts{ args, model };

    let mut state = tictactoe::State::default();

    let human_player = tictactoe::Player::O;

    loop {
        println!("{state}");

        if state.get_current_player() == human_player {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line).unwrap();
            let mut split = line.split_whitespace();
            let row = split.next().unwrap().parse::<usize>().unwrap();
            let col = split.next().unwrap().parse::<usize>().unwrap();

            let action = tictactoe::Action { row, col };
            state = state.get_next_state(&action).unwrap();
        } else {
            let action_probs = mcts.search(state.clone());
            println!("{action_probs:?}");
            let action = action_probs.get_best_action();
            state = state.get_next_state(&action).unwrap();
        }

        let (value, is_terminal) = state.get_value_and_terminated();
        if is_terminal {
            println!("{state}");
            if value == 0.0 {
                println!("Draw");
            } else {
                let winner = state.get_current_player().get_opposite();
                println!("Winner: {winner}");
            }
            break;
        }
    }
}

fn main() {
    set_print_options_short();

    // train();
    
    play();
}
