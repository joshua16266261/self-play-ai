mod model;
mod game;
mod mcts;
mod learner;

use model::{Model};
use mcts::{Args, Tree, Mcts};
use learner::Learner;
use tch::{nn::VarStore, Device};
use game::{State, Policy, Status, Player};
use chess::ChessMove;

// static CHECKPOINT_DIR: &str = "tictactoe_cp";
// static HUMAN_PLAYER: game::tictactoe::Player = game::tictactoe::Player::X;
static CHECKPOINT_DIR: &str = "chess_cp";
static HUMAN_PLAYER: game::chess::Player = game::chess::Player::White;

fn train() {
    let mut var_store = VarStore::new(Device::Cpu);
    let net = model::chess::Net::new(
        &var_store.root(),
        10,
        256
    );
    var_store.set_device(Device::Mps);

    let args = Args {
        num_learn_iters: 2, // Just for profiling
        num_searches: 60,
        num_self_play_iters: 10,
        // num_parallel_self_play_games: 64, 
        num_parallel_self_play_games: 2,
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
    // let net = model::tictactoe::Net::new(&var_store.root(), 4, 64);
    let net = model::chess::Net::new(
        &var_store.root(), 
        10, 
        256
    );
    var_store.load(format!("{CHECKPOINT_DIR}/2.safetensors")).unwrap();
    var_store.set_device(Device::Mps);

    // let args = Args::default();
    let args = Args {
        num_searches: 6000,
        ..Default::default()
    };
    let model = Model{ args, net };
    let mcts = Mcts{ args, model };

    let mut state = game::chess::State::default();
    let mut tree = Tree::with_root_state(state.clone());
    // let mut tree_vec = ;

    loop {
        println!("{state}");

        let action =
            if state.get_current_player() == HUMAN_PLAYER {
                // let mut line = String::new();
                // let _ = std::io::stdin().read_line(&mut line).unwrap();
                // let mut split = line.split_whitespace();
                // let row = split.next().unwrap().parse::<usize>().unwrap();
                // let col = split.next().unwrap().parse::<usize>().unwrap();

                // game::tictactoe::Action { row, col }

                let mut line = String::new();
                let _ = std::io::stdin().read_line(&mut line).unwrap();
                ChessMove::from_san(&state.game.current_position(), line.as_str()).unwrap()
            } else {
                let (_, child_id_to_probs) = &mcts.search(&mut vec![&mut tree])[0];

                let best_child_id = child_id_to_probs
                    .iter()
                    .max_by(|(_, prob1), (_, prob2)| prob1.total_cmp(prob2))
                    .map(|(id, _)| *id)
                    .unwrap();
                
                tree.use_subtree(best_child_id);

                tree.arena.get(0).unwrap().action_taken.unwrap()
                // search_results.get_best_action()
            };

        state = state.get_next_state(&action).unwrap();

        match state.get_status() {
            Status::Won => {
                println!("{state}");
                println!("Winner: {:?}", state.get_current_player().get_opposite());
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
    train();
    
    // play();
}
