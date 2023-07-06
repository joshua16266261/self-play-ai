use std::default::Default;
use std::fmt;

#[derive(Clone, strum_macros::Display, Default)]
enum Player {
    #[default]
    X,
    O
}

#[derive(Default, Clone)]
struct Piece(Option<Player>);

#[derive(Default, Clone)]
struct Board([[Piece; 3]; 3]);

#[derive(Default, Clone)]
pub struct State {
    board: Board,
    current_player: Player
}

pub struct Action {
    pub row: usize,
    pub col: usize
}

impl Player {
    fn get_opposite_player(&self) -> Self {
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

impl Board {
    fn get_piece(&self, row: usize, col: usize) -> &Piece {
        &self.0[row][col]
    }

    fn set_piece(&mut self, row: usize, col: usize, player: Player) {
        self.0[row][col] = Piece(Some(player));
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Current player: {}\n{}", self.current_player, self.board)
    }
}

impl State {
    pub fn get_next_state(&self, action: &Action) -> Result<Self, String> {
        return match &self.board.get_piece(action.row, action.col).0 {
            Some(player) => Err(format!("Player {player} has already played on that square")),
            None => {
                let mut next_state = self.clone();
                next_state.board.set_piece(action.row, action.col, self.current_player.clone());
                next_state.current_player = Player::get_opposite_player(&self.current_player);
                Ok(next_state)
            }
        };
    }
}