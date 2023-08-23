use std::{fmt, cmp};
use rand::distributions::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::prelude::*;
use ndarray::{Array, Array1, Array3, ArrayView1};

#[derive(Clone, Copy, Debug, strum_macros::Display, Default, PartialEq, Eq)]
pub enum Player {
    #[default]
    X,
    O
}

#[derive(Default, Clone, PartialEq, Eq, Debug)]
struct Piece(Option<Player>);

#[derive(Default, Clone, Debug, PartialEq)]
struct Board([[Piece; 7]; 6]);

#[derive(Default, Clone, PartialEq)]
pub struct State {
    board: Board,
    current_player: Player,
    num_actions_played: u8,
    status: super::Status
}

#[derive(Clone, Debug)]
pub struct Action(pub usize);

#[derive(Clone, Debug)]
pub struct Policy(Array1<f32>);

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
            " {} | {} | {} | {} | {} | {}| {}\n {} | {} | {} | {} | {} | {}| {}\n {} | {} | {} | {} | {} | {}| {}\n {} | {} | {} | {} | {} | {}| {}\n {} | {} | {} | {} | {} | {}| {}\n {} | {} | {} | {} | {} | {}| {}",
            self.0[5][0], self.0[5][1], self.0[5][2], self.0[5][3], self.0[5][4], self.0[5][5], self.0[5][6],
            self.0[4][0], self.0[4][1], self.0[4][2], self.0[4][3], self.0[4][4], self.0[4][5], self.0[4][6],
            self.0[3][0], self.0[3][1], self.0[3][2], self.0[3][3], self.0[3][4], self.0[3][5], self.0[3][6],
            self.0[2][0], self.0[2][1], self.0[2][2], self.0[2][3], self.0[2][4], self.0[2][5], self.0[2][6],
            self.0[1][0], self.0[1][1], self.0[1][2], self.0[1][3], self.0[1][4], self.0[1][5], self.0[1][6],
            self.0[0][0], self.0[0][1], self.0[0][2], self.0[0][3], self.0[0][4], self.0[0][5], self.0[0][6],
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
        write!(f, "Row: {}", self.0)
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self(Array::zeros(7))
    }
}

impl super::Policy for Policy {
    type Action = Action;

    fn get_prob(&self, action: &Action) -> f32 {
        self.0[[action.0]]
    }

    fn set_prob(&mut self, action: &Action, prob: f32) {
        self.0[[action.0]] = prob;
    }

    fn normalize(&mut self) {
        self.0 /= self.0.sum();
    }

    fn get_flat_ndarray(&self) -> Array1<f32> {
        self.0.clone()
    }

    fn sample(&self, rng: &mut ThreadRng, temperature: f32) -> Action {
        // Higher temperature => squishes probabilities together => encourages more exploration
        // let temperature_action_probs: Vec<f32> = self.0
        //     .iter()
        //     .map(|x| f32::powf(*x, temperature))
        //     .collect();
        let temperature_action_probs = self.0.mapv(|x| x.powf(temperature));
        let dist = WeightedIndex::new(temperature_action_probs).unwrap();
        let idx = dist.sample(rng);
        Action(idx)
    }

    fn get_best_action(&self) -> Action {
        let best_action_idx = self.0
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        Action(best_action_idx)
    }
}

impl Board {
    fn get_next_row_idx(&self, col_idx: usize) -> Option<usize> {
        // for i in 0..self.0.len() {
        //     if self.0[i][col_idx].0.is_none() {
        //         return Some(i);
        //     }
        // }
        // None
        (0..self.0.len()).find(|&i| self.0[i][col_idx].0.is_none())
    }
}

impl State {
    fn get_winner(&self, latest_row: usize, latest_col: usize) -> Option<Player> {
        // Check row win
        let row = &self.board.0[latest_row];
        for i in 0..=row.len() - 4 {
            if row[i].0.is_some() && row[i] == row[i + 1] &&
                row[i] == row[i + 2] &&
                row[i] == row[i + 3] {

                return row[i].0
            }
        }

        // Check col win
        for i in 0..=self.board.0.len() - 4 {
            if self.board.0[i][latest_col].0.is_some() &&
                self.board.0[i][latest_col] == self.board.0[i + 1][latest_col] &&
                self.board.0[i][latest_col] == self.board.0[i + 2][latest_col] &&
                self.board.0[i][latest_col] == self.board.0[i + 3][latest_col] {

                return self.board.0[i][latest_col].0
            }
        }

        // Check diagonal win
        let start_offset = cmp::max(-4, -(cmp::min(latest_col, latest_row) as i32));
        let end_offset = cmp::min(0, cmp::min(row.len() as i32 - (latest_col as i32 + 4), self.board.0.len() as i32 - (latest_row as i32 + 4)));
        for i in start_offset..=end_offset {
            let row = (latest_row as i32 + i) as usize;
            let col = (latest_col as i32 + i) as usize;
            if self.board.0[row][col].0.is_some() &&
                self.board.0[row][col] == self.board.0[row + 1][col + 1] &&
                self.board.0[row][col] == self.board.0[row + 2][col + 2] &&
                self.board.0[row][col] == self.board.0[row + 3][col + 3] {
                
                return self.board.0[row][col].0
            }
        }

        None
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
            super::Status::Ongoing => match self.board.get_next_row_idx(action.0) {
                None => Err("Illegal move: column already filled".to_string()),
                Some(row_idx) => {
                    let mut next_state = self.clone();
                    next_state.board.0[row_idx][action.0] = Piece(Some(self.current_player));
                    next_state.current_player = super::Player::get_opposite(&self.current_player);
                    next_state.num_actions_played += 1;

                    if next_state.get_winner(row_idx, action.0).is_some() {
                        next_state.status = super::Status::Won;
                    } else if next_state.num_actions_played == 6 * 7 {
                        next_state.status = super::Status::Tied;
                    }

                    Ok(next_state)
                }
            },
            _ => Err("Game has already ended".to_string())
        }        
    }

    fn get_valid_actions(&self) -> Vec<Action> {
        if self.status == super::Status::Ongoing {
            let mut valid_actions: Vec<Action> = Vec::with_capacity(7);
            let last_row = self.board.0.last().unwrap();
            for col in 0..7 {
                if last_row[col].0.is_none() {
                    valid_actions.push(Action(col));
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
        let mut encoding = Array3::zeros((3, 6, 7));
        for row in 0..6 {
            for col in 0..7 {
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
        if policy.shape() != [7,] {
            return Err(format!("Expected policy shape to be (7,), found {:?}", policy.shape()));
        }

        let policy = policy.into_shape(7).unwrap();

        let valid_actions = self.get_valid_actions();

        let mut mask = Array::zeros(7);
        for action in valid_actions {
            mask[action.0] = 1.0;
        }

        let mut masked_policy = &policy * &mask;
        masked_policy /= masked_policy.sum();

        Ok(Policy(masked_policy))
    }

    fn get_zero_policy(&self) -> Policy {
        Policy::default()
    }
}