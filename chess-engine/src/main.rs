mod tictactoe;
mod mcts;
mod model;
use model::{Model, TicTacToeNet};
use mcts::{Args, Learner, MCTS};
use tch::{nn, Device};
use tch::display::set_print_options_short;
use tictactoe::{State, Player, Action};

fn train() {
    let mut var_store = nn::VarStore::new(Device::Cpu);
    // let net = tch::TrainableCModule::load("../tictactoe_model.pt", var_store.root()).unwrap();
    let net = TicTacToeNet::new(&var_store.root(), 4, 64);
    var_store.set_device(Device::Mps);

    // let args = Args {
    //     num_epochs: 10,
    //     num_learn_iters: 1,
    //     num_self_play_iters: 1,
    //     ..Default::default()
    // };
    let args = Args::default();
    let model = Model{ args, net };
    let mcts = MCTS{ args, model };

    let mut learner = Learner{
        args,
        mcts,
        var_store: &mut var_store
    };

    learner.learn();
}

fn play() {
    let mut var_store = nn::VarStore::new(Device::Cpu);
    let net = TicTacToeNet::new(&var_store.root(), 4, 64);
    var_store.load("../tictactoe_model_2.safetensors").unwrap();
    var_store.set_device(Device::Mps);

    let args = Args::default();
    let model = Model{ args, net };
    let mut mcts = MCTS{ args, model };

    let mut state = State::default();

    loop {
        println!("{state}");

        if state.current_player == Player::X {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line).unwrap();
            let mut split = line.split_whitespace();
            let row = split.next().unwrap().parse::<usize>().unwrap();
            let col = split.next().unwrap().parse::<usize>().unwrap();

            let action = Action { row, col };
            state = state.get_next_state(&action).unwrap();
        } else {
            let action_probs = mcts.search(state.clone());
            println!("{action_probs:?}");
            let best_action_idx = action_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            let action = Action { row: best_action_idx / 3, col: best_action_idx % 3 };
            state = state.get_next_state(&action).unwrap();
        }

        let (value, is_terminal) = state.get_value_and_terminated();
        if is_terminal {
            println!("{state}");
            if value == 0.0 {
                println!("Draw");
            } else {
                let winner = Player::get_opposite_player(&state.current_player);
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
