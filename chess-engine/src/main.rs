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
// use chess::ChessMove;
use learner_concurrent::{SelfPlayArgs, TrainingArgs, SelfPlayWorker, ModelTrainerWorker};
use ringbuf::{HeapRb, Rb, Consumer};
use std::{sync::{Arc, Mutex, RwLock, Condvar}, fs::create_dir};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// static CHECKPOINT_DIR: &str = "tictactoe_cp";
// static HUMAN_PLAYER: game::tictactoe::Player = game::tictactoe::Player::X;
// static CHECKPOINT_DIR: &str = "chess_cp";
// static HUMAN_PLAYER: game::chess::Player = game::chess::Player::White;
static CHECKPOINT_DIR: &str = "connect4_cp";
static HUMAN_PLAYER: game::connect_four::Player = game::connect_four::Player::X;

fn train() {
    let mut var_store = VarStore::new(Device::Cpu);
    let net: model::connect_four::Net = model::Net::new(
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
    let net: model::connect_four::Net = model::Net::new(
        &var_store.root(),
        Default::default()
    );

    var_store.load(format!("{CHECKPOINT_DIR}/9.safetensors")).unwrap();
    var_store.set_device(Device::Mps);

    // let args = Args::default();
    let args = Args {
        num_searches: 1000,
        ..Default::default()
    };
    let model = Model{ args, net };
    let mcts = Mcts{ args, model };

    let mut state = game::connect_four::State::default();
    let mut tree = Tree::with_root_state(state.clone());

    loop {
        println!("{state}");

        let action =
            if state.get_current_player() == HUMAN_PLAYER {
                let mut line = String::new();
                let _ = std::io::stdin().read_line(&mut line).unwrap();

                let col = line.split_whitespace().next().unwrap().parse::<usize>().unwrap();
                let action = game::connect_four::Action(col);

                // let mut split = line.split_whitespace();
                // let row = split.next().unwrap().parse::<usize>().unwrap();
                // let col = split.next().unwrap().parse::<usize>().unwrap();
                // let action = game::tictactoe::Action { row, col };

                let next_state = state.get_next_state(&action).unwrap();
                let child_id = tree.arena.iter().position(|node| node.state == next_state);

                if let Some(id) = child_id {
                    tree.use_subtree(id);
                } else {
                    tree = Tree::with_root_state(next_state);
                }

                action                

                // let mut line = String::new();
                // let _ = std::io::stdin().read_line(&mut line).unwrap();
                // ChessMove::from_san(&state.game.current_position(), line.as_str()).unwrap()
            } else {
                let (_, child_id_to_probs) = &mcts.search(&mut vec![&mut tree])[0];

                let best_child_id = child_id_to_probs
                    .iter()
                    .max_by(|(_, prob1), (_, prob2)| prob1.total_cmp(prob2))
                    .map(|(id, _)| *id)
                    .unwrap();
                
                tree.use_subtree(best_child_id);

                tree.arena.get(0).unwrap().action_taken.clone().unwrap()
                // search_results.get_best_action()
            };
        println!("{}", action);
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
    // let self_play_args = SelfPlayArgs {
    //     num_mcts_searches: 600,
    //     ..Default::default()
    // };
    let self_play_args: SelfPlayArgs = Default::default();
    // let training_args = TrainingArgs {
    //     num_batches_per_iter: 2,
    //     num_train_iters: 2,
    //     ..Default::default()
    // };
    let training_args: TrainingArgs = Default::default();
    // let args = Args {
    //     num_learn_iters: 2, // Just for profiling
    //     num_searches: self_play_args.num_mcts_searches,
    //     num_self_play_iters: 500,
    //     num_parallel_self_play_games: 2,
    //     ..Default::default()
    // };
    let args = Default::default();

    let replay_buffer_capacity = training_args.batch_size * 100;
    let replay_buffer = Arc::new((Mutex::new(HeapRb::new(replay_buffer_capacity)), Condvar::new()));

    let mut var_store = VarStore::new(Device::Cpu);
    let _: model::connect_four::Net = model::Net::new(&var_store.root(), Default::default());
    var_store.set_device(Device::Mps);

    // Model trainer worker
    let mut trainer_var_store = VarStore::new(Device::Cpu);
    trainer_var_store.copy(&var_store).unwrap();
    let trainer_net: model::connect_four::Net = model::Net::new(
        &trainer_var_store.root(),
        Default::default()
    );
    trainer_var_store.set_device(Device::Mps);
    
    let num_train_iters = training_args.num_train_iters;
    let optimizer = Adam::default().build(&var_store, 1e-3).unwrap();
    let mut model_trainer_worker = ModelTrainerWorker {
        args: training_args,
        replay_buffer: Arc::clone(&replay_buffer),
        net: trainer_net,
        var_store: trainer_var_store,
        optimizer
    };

    // Self-play workers
    let self_play_workers: Vec<_> = (0..6).map(|_| {
        let mut self_play_var_store = VarStore::new(Device::Cpu);
        self_play_var_store.copy(&var_store).unwrap();
        let self_play_net: model::connect_four::Net = model::Net::new(
            &self_play_var_store.root(),
            Default::default()
        );
        self_play_var_store.set_device(Device::Mps);
        self_play_var_store.freeze();

        let model = Model { args, net: self_play_net };
        let mcts = Mcts { args, model };
        SelfPlayWorker {
            args: self_play_args.clone(),
            replay_buffer: Arc::clone(&replay_buffer),
            mcts
        }
    }).collect();

    let varstore_rwlock = Arc::new(RwLock::new(var_store));
    let in_training_rwlock = Arc::new(RwLock::new(true));

    // Progress bars
    let multi_pb = MultiProgress::new();

    let replay_buffer_pb_style = ProgressStyle::with_template(
        "{prefix} {bar:40} {pos:>3}/{len:3}"
    )
        .unwrap()
        .progress_chars("##-");
    let replay_buffer_pb = multi_pb.add(ProgressBar::new(replay_buffer_capacity as u64));
    replay_buffer_pb.set_style(replay_buffer_pb_style);
    replay_buffer_pb.set_prefix("Replay buffer");
    let replay_buffer_pb_mutex = Arc::new(Mutex::new(replay_buffer_pb));

    let training_pb_style = ProgressStyle::with_template(
        "{prefix} [{elapsed_precise}] {bar:40} {pos:>3}/{len:3} {msg}"
    )
        .unwrap()
        .progress_chars("##-");
    let training_pb = multi_pb.add(ProgressBar::new(num_train_iters as u64));
    training_pb.set_style(training_pb_style);
    training_pb.set_prefix("Training");

    let self_play_pb_style = ProgressStyle::with_template(
        "{prefix} {spinner:.green}"
    )
        .unwrap()
        .tick_chars(r"/-\| ");

    // Spawn workers
    rayon::scope(|s| {
        for (i, mut worker) in self_play_workers.into_iter().enumerate() {
            let worker_checkpoint_path_rwlock = varstore_rwlock.clone();
            let worker_in_training_rwlock = in_training_rwlock.clone();

            let worker_pb = multi_pb.add(ProgressBar::new_spinner());
            worker_pb.set_style(self_play_pb_style.clone());
            worker_pb.set_prefix(format!("Self-play {i}"));

            let replay_buffer_pb = replay_buffer_pb_mutex.clone();

            s.spawn(move |_| worker.self_play_loop(worker_checkpoint_path_rwlock, worker_in_training_rwlock, &worker_pb, replay_buffer_pb));
        }
        s.spawn(|_| model_trainer_worker.train_loop(varstore_rwlock, in_training_rwlock, CHECKPOINT_DIR, &training_pb, replay_buffer_pb_mutex));
    })
}

fn main() {
    let _ = create_dir(CHECKPOINT_DIR);
    // train();

    // train_concurrent();

    play();
}
