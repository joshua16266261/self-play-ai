mod model;
mod game;
mod mcts;
mod learner;
mod learner_concurrent;

use model::{Model};
use mcts::{Args, Tree, Mcts};
use learner::Learner;
use tch::{nn::{VarStore, Adam, OptimizerConfig, Optimizer}, Device};
use game::{State, Policy, Status, Player};
use chess::ChessMove;
use learner_concurrent::{SelfPlayArgs, TrainingArgs, SelfPlayWorker, ModelTrainerWorker};
use ringbuf::{HeapRb, Rb, Consumer};
use std::sync::{Arc, Mutex, RwLock};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// static CHECKPOINT_DIR: &str = "tictactoe_cp";
// static HUMAN_PLAYER: game::tictactoe::Player = game::tictactoe::Player::X;
static CHECKPOINT_DIR: &str = "chess_cp";
static HUMAN_PLAYER: game::chess::Player = game::chess::Player::White;

fn train() {
    let mut var_store = VarStore::new(Device::Cpu);
    let net: model::chess::Net = model::Net::new(
        &var_store.root(),
        Default::default()
    );
    var_store.set_device(Device::Mps);

    // Just for profiling
    let args = Args {
        num_learn_iters: 2,
        num_searches: 2,
        num_self_play_iters: 2,
        // num_parallel_self_play_games: 64, 
        num_parallel_self_play_games: 1,
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
    let net: model::chess::Net = model::Net::new(
        &var_store.root(),
        Default::default()
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

fn train_concurrent() {
    // Args
    let self_play_args = SelfPlayArgs {
        num_mcts_searches: 2,
        ..Default::default()
    };
    let training_args = TrainingArgs::default();
    let args = Args {
        num_learn_iters: 2, // Just for profiling
        num_searches: self_play_args.num_mcts_searches,
        num_self_play_iters: 2,
        // num_parallel_self_play_games: 64, 
        num_parallel_self_play_games: 2,
        ..Default::default()
    };

    // Self-play workers
    let mut var_store = VarStore::new(Device::Cpu);
    let net: model::chess::Net = model::Net::new(
        &var_store.root(),
        Default::default()
    );
    var_store.set_device(Device::Mps);

    let model = Model{ args, net };
    let mcts = Mcts{ args, model };

    let replay_buffer = Arc::new(Mutex::new(HeapRb::new(640)));
    let checkpoint_path_rwlock = Arc::new(RwLock::new(None));

    let mut self_play_workers = [
        SelfPlayWorker {
            args: self_play_args,
            replay_buffer: Arc::clone(&replay_buffer),
            mcts
        }
    ];

    // Model trainer worker
    let mut trainer_var_store = VarStore::new(Device::Cpu);
    let trainer_net: model::chess::Net = model::Net::new(
        &trainer_var_store.root(),
        Default::default()
    );
    trainer_var_store.set_device(Device::Mps);
    
    let num_train_iters = training_args.num_train_iters;
    let optimizer = Adam::default().build(&trainer_var_store, 1e-3).unwrap();
    let mut model_trainer_worker = ModelTrainerWorker {
        args: training_args,
        replay_buffer: Arc::clone(&replay_buffer),
        net: trainer_net,
        var_store: trainer_var_store,
        optimizer
    };

    // Progress bars
    let pb_style = ProgressStyle::with_template(
        "{prefix:>9} [{elapsed_precise}] {bar:40} {pos:>7}/{len:7} (ETA: {eta_precise}) {msg}"
    )
        .unwrap()
        .progress_chars("##-");
    let pb = ProgressBar::new(num_train_iters as u64);
    pb.set_style(pb_style);
    pb.set_prefix("Training");

    // Spawn workers
    // model_trainer_worker.train_loop(checkpoint_path_rwlock, CHECKPOINT_DIR, &pb);
    // self_play_workers[0].self_play_loop(checkpoint_path_rwlock);

    rayon::scope(|s| {
        for mut worker in self_play_workers {
            let worker_checkpoint_path_rwlock = checkpoint_path_rwlock.clone();
            rayon::spawn(move || {
                worker.self_play_loop(worker_checkpoint_path_rwlock);
            });
        }
        s.spawn(|_| model_trainer_worker.train_loop(checkpoint_path_rwlock, CHECKPOINT_DIR, &pb));
    })
}

fn main() {
    // FIXME: Loss is NaN
    train();
    
    // play();

    // train_concurrent();
}
