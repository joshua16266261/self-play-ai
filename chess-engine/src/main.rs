mod model;
mod game;
mod mcts;
mod learner;

use model::{Model};
use mcts::{Args, Tree, Mcts};
use learner::Learner;
use tch::{nn::VarStore, Device};
use game::{State, Policy, Status};
use chess::ChessMove;

use crate::game::Player;

static CHECKPOINT_DIR: &str = "tictactoe_cp";
static HUMAN_PLAYER: game::tictactoe::Player = game::tictactoe::Player::X;

fn train() {
    let mut var_store = VarStore::new(Device::Cpu);
    let net = model::tictactoe::Net::new(
        &var_store.root(),
        4,
        64
    );
    var_store.set_device(Device::Mps);

    let args = Args {
        num_searches: 60,
        num_self_play_iters: 10,
        num_parallel_self_play_games: 50, 
        ..Default::default()
    };
    let model = Model{ args, net };
    let mcts = Mcts{ args, model };

    let learner = Learner{
        args,
        mcts,
        var_store: &mut var_store
    };

    learner.learn(CHECKPOINT_DIR);
}

fn play() {
    let mut var_store = VarStore::new(Device::Cpu);
    let net = model::tictactoe::Net::new(&var_store.root(), 4, 64);
    var_store.load(format!("{CHECKPOINT_DIR}/2.safetensors")).unwrap();
    var_store.set_device(Device::Mps);

    let args = Args::default();
    let model = Model{ args, net };
    let mcts = Mcts{ args, model };

    let mut state = game::tictactoe::State::default();

    loop {
        println!("{state}");

        let action =
            if state.get_current_player() == HUMAN_PLAYER {
                let mut line = String::new();
                let _ = std::io::stdin().read_line(&mut line).unwrap();
                let mut split = line.split_whitespace();
                let row = split.next().unwrap().parse::<usize>().unwrap();
                let col = split.next().unwrap().parse::<usize>().unwrap();

                game::tictactoe::Action { row, col }
            } else {
                mcts.search(&mut vec![Tree::with_root_state(state.clone())])[0].get_best_action()
            };

        state = state.get_next_state(&action).unwrap();

        match state.get_status() {
            Status::Won => {
                println!("{state}");
                println!("Winner: {}", state.get_current_player().get_opposite());
                break;
            },
            Status::Tied => {
                println!("{state}");
                println!("Draw");
                break;
            },
            Status::Ongoing => ()
        };
    }
}

fn main() {
    // train();
    
    // play();

    // Test chess
    let mut state = game::chess::State::default();

    loop {
        match state.get_status() {
            Status::Ongoing => {
                println!("{}", state);
    
                let mut line = String::new();
                let _ = std::io::stdin().read_line(&mut line).unwrap();
                let action = ChessMove::from_san(&state.game.current_position(), line.as_str()).unwrap();
    
                state = state.get_next_state(&action).unwrap();
            },
            Status::Won => {
                println!("{}", state);
                println!("Winner: {:?}", state.get_current_player().get_opposite());
                break;
            },
            Status::Tied => {
                println!("{}", state);
                println!("Draw");
                break;
            },
        }
    }
    
}
