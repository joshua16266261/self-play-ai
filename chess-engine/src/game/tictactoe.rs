use std::fmt;
use rand::distributions::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::prelude::*;
use ndarray::{Array, Array1, Array2, Array3, ArrayView1};

#[derive(Clone, Copy, Debug, strum_macros::Display, Default, PartialEq, Eq)]
pub enum Player {
    #[default]
    X,
    O
}

#[derive(Default, Clone, PartialEq, Eq, Debug)]
struct Piece(Option<Player>);

#[derive(Default, Clone, Debug)]
struct Board([[Piece; 3]; 3]);

#[derive(Default, Clone)]
pub struct State {
    board: Board,
    current_player: Player,
    num_actions_played: u8,
    status: super::Status
}

#[derive(Clone, Debug)]
pub struct Action {
    pub row: usize,
    pub col: usize
}

#[derive(Clone)]
pub struct Policy(Array2<f32>);

impl super::Player for Player {
    fn get_opposite(&self) -> Self {
        match self {
            Player::X => Player::O,
            Player::O => Player::X
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match &self.0 {
            Some(player) => player.to_string(),
            None => "-".to_string()
        })
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            " {} | {} | {}\n {} | {} | {}\n {} | {} | {}",
            self.0[0][0], self.0[0][1], self.0[0][2],
            self.0[1][0], self.0[1][1], self.0[1][2],
            self.0[2][0], self.0[2][1], self.0[2][2]
        )
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Current player: {}\n{}", self.current_player, self.board)
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self(Array::zeros((3, 3)))
    }
}

impl super::Policy for Policy {
    type Action = Action;

    fn get_prob(&self, action: &Action) -> f32 {
        self.0[[action.row, action.col]]
    }

    fn set_prob(&mut self, action: &Action, prob: f32) {
        self.0[[action.row, action.col]] = prob;
    }

    fn normalize(&mut self) {
        self.0 /= self.0.sum();
    }

    fn get_flat_ndarray(&self) -> Array1<f32> {
        self.0.clone().into_shape((9,)).unwrap()
    }

    fn sample(&self, rng: &mut ThreadRng, temperature: f32) -> Action {
        // Higher temperature => squishes probabilities together => encourages more exploration
        // let temperature_action_probs: Vec<f32> = self.0
        //     .iter()
        //     .map(|x| f32::powf(*x, temperature))
        //     .collect();
        let temperature_action_probs = self.get_flat_ndarray().mapv(|x| x.powf(temperature));
        let dist = WeightedIndex::new(temperature_action_probs).unwrap();
        let idx = dist.sample(rng);
        Action{ row: idx / 3, col: idx % 3}
    }

    fn get_best_action(&self) -> Action {
        let best_action_idx = self.0
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        Action { row: best_action_idx / 3, col: best_action_idx % 3 }
    }
}

impl super::State for State {
    type Policy = Policy;
    type Player = Player;

    fn get_current_player(&self) -> Self::Player {
        self.current_player
    }

    fn get_next_state(&self, action: &Action) -> Result<Self, String> {
        match self.status {
            super::Status::Ongoing => match &self.board.0[action.row][action.col].0 {
                Some(player) => Err(format!("Illegal move: player {player} has already played on that square")),
                None => {
                    let mut next_state = self.clone();
                    next_state.board.0[action.row][action.col] = Piece(Some(self.current_player));
                    next_state.current_player = super::Player::get_opposite(&self.current_player);
                    next_state.num_actions_played += 1;
    
                    let row = &next_state.board.0[action.row];
                    let is_row_win = row[0] == row[1] && row[1] == row[2];
    
                    let is_col_win =
                        next_state.board.0[0][action.col] == next_state.board.0[1][action.col] &&
                            next_state.board.0[1][action.col] == next_state.board.0[2][action.col];
                    
                    let is_nw_se_diag_win = action.row == action.col && 
                        next_state.board.0[0][0] == next_state.board.0[1][1] && next_state.board.0[1][1] == next_state.board.0[2][2];
                    let is_ne_sw_diag_win = ((action.row == 1 && action.col == 1) || (action.row).abs_diff(action.col) == 2) &&
                        next_state.board.0[0][2] == next_state.board.0[1][1] && next_state.board.0[1][1] == next_state.board.0[2][0];
                    
                    if is_row_win || is_col_win || is_nw_se_diag_win || is_ne_sw_diag_win {
                        next_state.status = super::Status::Won
                    } else if next_state.num_actions_played == 9 {
                        next_state.status = super::Status::Tied
                    }
                    Ok(next_state)
                }
            },
            _ => Err("Game has already ended".to_string())
        }        
    }

    fn get_valid_actions(&self) -> Vec<Action> {
        if self.status == super::Status::Ongoing {
            let mut valid_actions: Vec<Action> = Vec::with_capacity(9);
            for row in 0..3 {
                for col in 0..3 {
                    if self.board.0[row][col].0.is_none() {
                        valid_actions.push(Action { row, col });
                    }
                }
            }
            return valid_actions;
        }
        Vec::with_capacity(0)
    }

    fn get_status(&self) -> super::Status {
        self.status
    }

    fn get_value_and_terminated(&self) -> (f32, bool) {
        // The value is given from the node's own perspective
        // i.e., if it's X to play but the game is over
        // then X is lost and the value is -1 (which is +1 from the parent's perspective)
        match self.status {
            super::Status::Won => (-1.0, true),
            super::Status::Tied => (0.0, true),
            super::Status::Ongoing => (0.0, false)
        }
    }

    fn get_encoding(&self) -> Array3<f32> {
        let mut encoding = Array3::zeros((3, 3, 3));
        for row in 0..3 {
            for col in 0..3 {
                match &self.board.0[row][col].0 {
                    Some(player) => {
                        if *player == self.current_player {
                            encoding[[0, row, col]] = 1.0;
                        } else {
                            encoding[[1, row, col]] = 1.0;
                        }
                    },
                    None => encoding[[2, row, col]] = 1.0
                };
            }
        }
        encoding
    }

    fn mask_invalid_actions(&self, policy: ArrayView1<f32>) -> Result<Policy, String> {
        if policy.shape() != [9,] {
            return Err(format!("Expected policy shape to be (9,), found {:?}", policy.shape()));
        }

        let policy = policy.into_shape((3, 3)).unwrap();

        let valid_actions = self.get_valid_actions();

        let mut mask = Array::zeros((3, 3));
        for action in valid_actions {
            mask[[action.row, action.col]] = 1.0;
        }

        let mut masked_policy = &policy * &mask;
        masked_policy /= masked_policy.sum();

        Ok(Policy(masked_policy))
    }
}
