use std::fmt;
use rand::distributions::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::prelude::*;

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

#[derive(Clone)]
pub struct Action {
    pub row: usize,
    pub col: usize
}

pub type Policy = [f32; 9];
pub type Encoding = [f32; 27];

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

impl super::Encoding for Encoding {
    fn get_flat_slice(&self) -> &[f32] {
        self.as_slice()
    }
}

impl super::Policy for Policy {
    type Action = Action;

    fn get_prob(&self, action: &Action) -> f32 {
        self[action.row * 3 + action.col]
    }

    fn set_prob(&mut self, action: &Action, prob: f32) {
        self[action.row * 3 + action.col] = prob;
    }

    fn normalize(&mut self) {
        let sum: f32 = self.iter().sum();
        self.map(|x| x / sum);
    }

    fn get_flat_slice(&self) -> &[f32] {
        self
    }

    fn sample(&self, rng: &mut ThreadRng, temperature: f32) -> Action {
        // Higher temperature => squishes probabilities together => encourages more exploration
        let temperature_action_probs: Vec<f32> = self
            .iter()
            .map(|x| f32::powf(*x, temperature))
            .collect();
        let dist = WeightedIndex::new(temperature_action_probs).unwrap();
        let idx = dist.sample(rng);
        Action{ row: idx / 3, col: idx % 3}
    }

    fn get_best_action(&self) -> Action {
        let best_action_idx = self
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        Action { row: best_action_idx / 3, col: best_action_idx % 3 }
    }
}

impl super::State for State {
    type Encoding = Encoding;
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
                        next_state.status = super::Status::Completed
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

    fn get_value_and_terminated(&self) -> (f32, bool) {
        // The value is given from the node's own perspective
        // i.e., if it's X to play but the game is over
        // then X is lost and the value is -1 (which is +1 from the parent's perspective)
        match self.status {
            super::Status::Completed => (-1.0, true),
            super::Status::Tied => (0.0, true),
            super::Status::Ongoing => (0.0, false)
        }
    }

    fn encode(&self) -> Encoding {
        let mut encoding: Encoding = [0.0; 27];
        for row in 0..3 {
            for col in 0..3 {
                match &self.board.0[row][col].0 {
                    Some(player) => {
                        if *player == self.current_player {
                            encoding[row * 3 + col] = 1.0;
                        } else {
                            encoding[9 + row * 3 + col] = 1.0;
                        }
                    },
                    None => encoding[18 + row * 3 + col] = 1.0
                };
            }
        }
        encoding
    }

    fn mask_invalid_actions(&self, policy: Vec<f32>) -> Result<Policy, String> {
        if policy.len() != 9 {
            return Err(format!("Expected policy length to be 9, found {}", policy.len()));
        }

        let valid_actions = self.get_valid_actions();

        let mut masked_policy: Policy = [0f32; 9];
        let mut total_prob = 0.0;
        for action in valid_actions {
            let idx = action.row * 3 + action.col;
            let prob = *policy.get(idx).unwrap();
            masked_policy[idx] = prob;
            total_prob += prob;
        }
        masked_policy = masked_policy.map(|x| x / total_prob);

        Ok(masked_policy)
    }
}