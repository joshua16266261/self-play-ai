use std::default::Default;
use std::fmt;

#[derive(Clone, strum_macros::Display, Default, PartialEq, Eq)]
pub enum Player {
    #[default]
    X,
    O
}

#[derive(Default, Clone, PartialEq, Eq)]
struct Piece(Option<Player>);

#[derive(Default, Clone)]
struct Board([[Piece; 3]; 3]);

#[derive(Clone, strum_macros::Display, Default, PartialEq, Eq)]
enum Status{
    #[default]
    Ongoing,
    Tied,
    Won
}

#[derive(Default, Clone)]
pub struct State {
    board: Board,
    pub current_player: Player,
    actions_played: u8,
    status: Status
}

pub struct Action {
    pub row: usize,
    pub col: usize
}

impl Player {
    pub fn get_opposite_player(&self) -> Self {
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

impl State {
    pub fn get_next_state(&self, action: &Action) -> Result<Self, String> {
        match self.status {
            Status::Ongoing => match &self.board.0[action.row][action.col].0 {
                Some(player) => Err(format!("Illegal move: player {player} has already played on that square")),
                None => {
                    let mut next_state = self.clone();
                    next_state.board.0[action.row][action.col] = Piece(Some(self.current_player.clone()));
                    next_state.current_player = Player::get_opposite_player(&self.current_player);
                    next_state.actions_played += 1;
    
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
                        next_state.status = Status::Won
                    } else if next_state.actions_played == 9 {
                        next_state.status = Status::Tied
                    }
                    Ok(next_state)
                }
            },
            _ => Err("Game has already ended".to_string())
        }        
    }

    pub fn get_valid_actions(&self) -> Vec<Action> {
        if self.status == Status::Ongoing {
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

    pub fn get_value_and_terminated(&self) -> (i32, bool) {
        match self.status {
            Status::Won => (1, true),
            Status::Tied => (0, true),
            Status::Ongoing => (0, false)
        }
    }

    pub fn encode(&self) -> [f32; 27] {
        let mut encoding = [0.0; 27];
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
}