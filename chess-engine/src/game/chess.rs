use chess::{Color, BoardStatus, Game, ChessMove, Piece, Square, Rank, File, MoveGen, Action};
use ndarray::{Array3, ArrayView1, Array, s};
use rand::distributions::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::prelude::*;
use std::fmt;

const NUM_PROMOTION_DIRS: usize = 3;
const NUM_SINGLE_SQUARE_STEPS: usize = 7;

const ROOK_PROMOTION_START_IDX: usize = 0;
const BISHOP_PROMOTION_START_IDX: usize = ROOK_PROMOTION_START_IDX + NUM_PROMOTION_DIRS;
const KNIGHT_PROMOTION_START_IDX: usize = BISHOP_PROMOTION_START_IDX + NUM_PROMOTION_DIRS;

const HORIZONTAL_MOVE_START_IDX: usize = KNIGHT_PROMOTION_START_IDX + NUM_PROMOTION_DIRS;
const VERTICAL_MOVE_START_IDX: usize = HORIZONTAL_MOVE_START_IDX + 2 * NUM_SINGLE_SQUARE_STEPS;
const DIAGONAL_MOVE_START_IDX: usize = VERTICAL_MOVE_START_IDX + 2 * NUM_SINGLE_SQUARE_STEPS;
const KNIGHT_MOVE_START_IDX: usize = DIAGONAL_MOVE_START_IDX + 4 * NUM_SINGLE_SQUARE_STEPS;

#[derive(Clone, Debug)]
pub struct Policy {
    probs: Array3<f32>,
    player: Color
}

#[derive(Clone)]
pub struct State {
    pub game: Game,
    transposition_table: Vec<Vec<ChessMove>>,
    fifty_move_rule_halfmove_counter: u64
}

pub type Player = Color;

impl super::Player for Player {
    fn get_opposite(&self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White
        }
    }
}

impl State {
    fn get_num_repetitions(&self) -> u64 {
        let mut counter = 0;
        let current_pos: Vec<ChessMove> = MoveGen::new_legal(&self.game.current_position()).collect();
        for pos in &self.transposition_table {
            if current_pos == *pos {
                counter += 1;
            }
        }
        counter + 1
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let board = self.game.current_position();
        let mut board_display = String::new();

        for row in 0..8 {
            board_display.push('|');
            for file in 0..8 {
                let square = Square::make_square(Rank::from_index(7 - row), File::from_index(file));
                let piece_str = match board.piece_on(square) {
                    Some(Piece::Pawn) => " p |",
                    Some(Piece::Knight) => " n |",
                    Some(Piece::Bishop) => " b |",
                    Some(Piece::Rook) => " r |",
                    Some(Piece::Queen) => " q |",
                    Some(Piece::King) => " k |",
                    None => " - |"
                };
                match board.color_on(square) {
                    Some(Color::White) => board_display.push_str(piece_str.to_uppercase().as_str()),
                    _ => board_display.push_str(piece_str),
                }
            }
            board_display.push('\n');
        }

        write!(f, "Current player: {:?}\n{}", self.game.side_to_move(), board_display)
    }
}

impl Default for State {
    fn default() -> Self {
        Self {
            game: Game::new(),
            transposition_table: Vec::new(),
            fifty_move_rule_halfmove_counter: 0
        }
    }
}

impl super::State for State {
    type Policy = Policy;
    type Player = Color;

    fn get_current_player(&self) -> Self::Player {
        self.game.current_position().side_to_move()
    }

    fn get_next_state(&self, action: &ChessMove) -> Result<Self, String> {
        if self.get_status() != super::Status::Ongoing {
            return Err("Game is already over".to_string());
        }

        let mut next_state = self.game.clone();
        let made_move = next_state.make_move(*action);

        if made_move {
            let mut next_transposition_table = self.transposition_table.clone();
            next_transposition_table.push(MoveGen::new_legal(&self.game.current_position()).collect());

            // Check if action is reversible
            let board = self.game.current_position();

            let white_castle_rights = board.castle_rights(Color::White);
            let black_castle_rights = board.castle_rights(Color::Black);

            let is_reversible = !board.piece_on(action.get_source()).is_some_and(|piece| piece == Piece::Pawn) &&
                board.piece_on(action.get_dest()).is_none() &&
                next_state.current_position().castle_rights(Color::White) == white_castle_rights &&
                next_state.current_position().castle_rights(Color::Black) == black_castle_rights;

            Ok(Self{
                game: next_state,
                transposition_table: next_transposition_table,
                fifty_move_rule_halfmove_counter:
                    if is_reversible {
                        self.fifty_move_rule_halfmove_counter + 1
                    } else {
                        0
                    }
            })
        } else {
            Err("Failed to make move".to_string())
        }
    }

    fn get_valid_actions(&self) -> Vec<ChessMove> {
        MoveGen::new_legal(&self.game.current_position()).collect()
    }

    fn get_status(&self) -> super::Status {
        match self.game.current_position().status() {
            BoardStatus::Checkmate => super::Status::Won,
            BoardStatus::Stalemate => super::Status::Tied,
            BoardStatus::Ongoing => {
                if self.get_num_repetitions() >= 3 || self.fifty_move_rule_halfmove_counter >= 100 {
                    super::Status::Tied
                } else {
                    super::Status::Ongoing
                }
            }
        }
    }

    fn get_value_and_terminated(&self) -> (f32, bool) {
        match self.get_status() {
            super::Status::Ongoing => (0.0, false),
            super::Status::Tied => (0.0, true),
            super::Status::Won => (1.0, true)
        }
    }

    fn get_encoding(&self) -> Array3<f32> {
        let board = self.game.current_position();
        let current_player = self.get_current_player();

        let mut encoding = Array::zeros((19, 8, 8));

        // Piece positions
        for row in 0..8 {
            let rank = match current_player {
                Color::White => Rank::from_index(row),
                Color::Black => Rank::from_index(7 - row)
            };

            for col in 0..8 {
                let square = Square::make_square(rank, File::from_index(col));
                let color = board.color_on(square);
                let offset =
                    if let Some(player) = color {
                        if player == current_player {
                            0
                        } else {
                            6
                        }
                    } else {
                        continue
                    };

                match board.piece_on(square) {
                    Some(Piece::Pawn) => encoding[[offset, row, col]] = 1.0,
                    Some(Piece::Knight) => encoding[[offset + 1, row, col]] = 1.0,
                    Some(Piece::Bishop) => encoding[[offset + 2, row, col]] = 1.0,
                    Some(Piece::Rook) => encoding[[offset + 3, row, col]] = 1.0,
                    Some(Piece::Queen) => encoding[[offset + 4, row, col]] = 1.0,
                    Some(Piece::King) => encoding[[offset + 5, row, col]] = 1.0,
                    None => ()
                }
            }
        }

        // Castling rights
        let my_castle_rights = board.my_castle_rights();
        if my_castle_rights.has_kingside() {
            encoding.slice_mut(s![12, .., ..]).fill(1.0);
        }
        if my_castle_rights.has_queenside() {
            encoding.slice_mut(s![13, .., ..]).fill(1.0);
        }
        let their_castle_rights = board.their_castle_rights();
        if their_castle_rights.has_kingside() {
            encoding.slice_mut(s![14, .., ..]).fill(1.0);
        }
        if their_castle_rights.has_queenside() {
            encoding.slice_mut(s![15, .., ..]).fill(1.0);
        }

        // Number of occurences (for 3-fold repetition)
        encoding.slice_mut(s![16, .., ..]).fill(self.get_num_repetitions() as f32);

        // 50 move rule counter
        // TODO: Do we floor divide fifty_move_rule_halfmove_counter by 2?
        encoding.slice_mut(s![17, .., ..]).fill((self.fifty_move_rule_halfmove_counter as f32) / 100.0);
        
        // Total move counter
        // TODO: Is fullmove counter better or halfmove?
        let total_moves = self.game.actions()
            .iter()
            .filter(|action| matches!(action, Action::MakeMove(_)))
            .count() / 2;
        encoding.slice_mut(s![18, .., ..]).fill((total_moves as f32) / 50.0);

        

        encoding
    }

    fn mask_invalid_actions(&self, policy: ArrayView1<f32>) -> Result<Policy, String> {
        if policy.shape() != [73 * 8 * 8,] {
            return Err(format!("Expected policy shape to be (73 * 8 * 8,), found {:?}", policy.shape()));
        }

        let reshaped_policy = policy.into_shape((73, 8, 8)).unwrap().into_owned();
        let mut policy = Policy {
            probs: reshaped_policy,
            player: self.get_current_player()
        };

        let mut mask = Policy { player: policy.player, ..Default::default() };
        for action in MoveGen::new_legal(&self.game.current_position()) {
            super::Policy::set_prob(&mut mask, &action, 1.0);
        }

        let mut masked_probs = &policy.probs * &mask.probs;
        masked_probs /= masked_probs.sum();
        policy.probs = masked_probs;

        Ok(policy)

        // if policy.shape() != [73 * 8 * 8,] {
        //     return Err(format!("Expected policy shape to be (73 * 8 * 8,), found {:?}", policy.shape()));
        // }

        // let reshaped_policy = policy.into_shape((73, 8, 8)).unwrap().into_owned();
        // let mut policy = Policy {
        //     probs: reshaped_policy,
        //     player: self.get_current_player()
        // };

        // for action in MoveGen::new_legal(&self.game.current_position()) {
        //     // policy.set_prob(&action, 0.0);
        //     super::Policy::set_prob(&mut policy, &action, 0.0);
        // }

        // super::Policy::normalize(&mut policy);

        // Ok(policy)
    }

    fn get_zero_policy(&self) -> Policy {
        Policy {
            player: self.get_current_player(),
            ..Default::default()
        }
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self{
            probs: Array::zeros((73, 8, 8)),
            player: Color::White
        }
    }
}

impl Policy {
    fn get_channel(&self, action: &ChessMove) -> usize {
        // Convert a move to an int in [0, 73) representing the index in the policy.
        // let chess_move = action.chess_move;

        let source_square = action.get_source();
        let dest_square = action.get_dest();
        let mut rank_diff = dest_square.get_rank().to_index() as i64 - source_square.get_rank().to_index() as i64;
        let file_diff = dest_square.get_file().to_index() as i64 - source_square.get_file().to_index() as i64;
        let abs_rank_diff = rank_diff.abs();
        let abs_file_diff = file_diff.abs();

        if self.player == Color::Black {
            rank_diff *= -1;
        }

        // Under-promotion
        let sub_idx = (file_diff + 1) as usize; // Left: 0, Straight: 1, Right: 2
        match action.get_promotion() {
            Some(Piece::Rook) => return ROOK_PROMOTION_START_IDX + sub_idx,
            Some(Piece::Bishop) => return BISHOP_PROMOTION_START_IDX + sub_idx,
            Some(Piece::Knight) => return KNIGHT_PROMOTION_START_IDX + sub_idx,
            None | Some(Piece::Queen) | Some(Piece::King) | Some(Piece::Pawn) => ()
        }

        // Horizontal
        if rank_diff == 0 {
            if file_diff < 0 {
                return HORIZONTAL_MOVE_START_IDX + (-file_diff as usize) - 1;
            }
            return HORIZONTAL_MOVE_START_IDX + NUM_SINGLE_SQUARE_STEPS + (file_diff as usize) - 1;
        }

        // Vertical
        if file_diff == 0 {
            if rank_diff < 0 {
                return VERTICAL_MOVE_START_IDX + (-rank_diff as usize) - 1;
            }
            return VERTICAL_MOVE_START_IDX + NUM_SINGLE_SQUARE_STEPS + (rank_diff as usize) - 1;
        }

        // Diagonal
        if abs_rank_diff == abs_file_diff {
            if file_diff < 0 {
                if rank_diff > 0 {
                    return DIAGONAL_MOVE_START_IDX + (rank_diff as usize) - 1; // Northwest
                }
                return DIAGONAL_MOVE_START_IDX + NUM_SINGLE_SQUARE_STEPS + (-rank_diff as usize) - 1; // Southwest
            } else {
                if rank_diff > 0 {
                    return DIAGONAL_MOVE_START_IDX + 2 * NUM_SINGLE_SQUARE_STEPS + (rank_diff as usize) - 1; // Northeast
                }
                return DIAGONAL_MOVE_START_IDX + 3 * NUM_SINGLE_SQUARE_STEPS + (-rank_diff as usize) - 1; // Southeast 
            }
        }

        // Knight move
        if file_diff < 0 {
            if rank_diff > 0 { // Northwest
                if abs_rank_diff > abs_file_diff {
                    return KNIGHT_MOVE_START_IDX;
                }
                KNIGHT_MOVE_START_IDX + 1
            } else { // Southwest
                if abs_rank_diff > abs_file_diff {
                    return KNIGHT_MOVE_START_IDX + 2;
                }
                KNIGHT_MOVE_START_IDX + 3
            }
        } else if rank_diff > 0 { // Northeast
            if abs_rank_diff > abs_file_diff {
                return KNIGHT_MOVE_START_IDX + 4;
            }
            return KNIGHT_MOVE_START_IDX + 5;
        } else { // Southeast
            if abs_rank_diff > abs_file_diff {
                return KNIGHT_MOVE_START_IDX + 6;
            }
            return KNIGHT_MOVE_START_IDX + 7;
        }
    }

    fn get_action(&self, channel: usize, mut row: usize, col: usize) -> ChessMove {
        let promotion =
            if channel < BISHOP_PROMOTION_START_IDX {
                Some(Piece::Rook)
            } else if channel < KNIGHT_PROMOTION_START_IDX {
                Some(Piece::Bishop)
            } else if channel < HORIZONTAL_MOVE_START_IDX {
                Some(Piece::Knight)
            } else {
                None
            };

        let mut rank_diff: i64 =
            if channel < HORIZONTAL_MOVE_START_IDX {
                1
            } else if channel < VERTICAL_MOVE_START_IDX {
                0
            } else if channel < DIAGONAL_MOVE_START_IDX {
                let offset = channel - VERTICAL_MOVE_START_IDX;
                if offset < NUM_SINGLE_SQUARE_STEPS {
                    -((channel + 1 - VERTICAL_MOVE_START_IDX) as i64)
                } else {
                    (channel + 1 - VERTICAL_MOVE_START_IDX - NUM_SINGLE_SQUARE_STEPS) as i64
                }
            } else if channel < KNIGHT_MOVE_START_IDX {
                let offset = channel - DIAGONAL_MOVE_START_IDX;
                if offset < NUM_SINGLE_SQUARE_STEPS {
                    (channel + 1 - DIAGONAL_MOVE_START_IDX) as i64
                } else if offset < 2 * NUM_SINGLE_SQUARE_STEPS {
                    -((channel + 1 - DIAGONAL_MOVE_START_IDX - NUM_SINGLE_SQUARE_STEPS) as i64)
                } else if offset < 3 * NUM_SINGLE_SQUARE_STEPS {
                    (channel + 1 - DIAGONAL_MOVE_START_IDX - 2 * NUM_SINGLE_SQUARE_STEPS) as i64
                } else {
                    -((channel + 1 - DIAGONAL_MOVE_START_IDX - 3 * NUM_SINGLE_SQUARE_STEPS) as i64)
                } 
            } else {
                let offset = channel - KNIGHT_MOVE_START_IDX;
                match offset {
                    0 | 4 => 2,
                    1 | 5 => 1,
                    2 | 6 => -2,
                    3 | 7 => -1,
                    _ => unreachable!()
                }
            };

        let file_diff: i64 =
            if channel < HORIZONTAL_MOVE_START_IDX {
                if channel < BISHOP_PROMOTION_START_IDX {
                    (channel - 1) as i64
                } else if channel < KNIGHT_MOVE_START_IDX {
                    (channel - BISHOP_PROMOTION_START_IDX - 1) as i64
                } else {
                    (channel - KNIGHT_PROMOTION_START_IDX - 1) as i64
                }
            } else if channel < VERTICAL_MOVE_START_IDX {
                let offset = channel - HORIZONTAL_MOVE_START_IDX;
                if offset < NUM_SINGLE_SQUARE_STEPS {
                    -((channel + 1 - HORIZONTAL_MOVE_START_IDX) as i64)
                } else {
                    (channel + 1 - HORIZONTAL_MOVE_START_IDX - NUM_SINGLE_SQUARE_STEPS) as i64
                }
            } else if channel < DIAGONAL_MOVE_START_IDX {
                0
            } else if channel < KNIGHT_MOVE_START_IDX {
                let offset = channel - DIAGONAL_MOVE_START_IDX;
                if offset < NUM_SINGLE_SQUARE_STEPS {
                    -((channel + 1 - DIAGONAL_MOVE_START_IDX) as i64)
                } else if offset < 2 * NUM_SINGLE_SQUARE_STEPS {
                    -((channel + 1 - DIAGONAL_MOVE_START_IDX - NUM_SINGLE_SQUARE_STEPS) as i64)
                } else if offset < 3 * NUM_SINGLE_SQUARE_STEPS {
                    (channel + 1 - DIAGONAL_MOVE_START_IDX - 2 * NUM_SINGLE_SQUARE_STEPS) as i64
                } else {
                    (channel + 1 - DIAGONAL_MOVE_START_IDX - 3 * NUM_SINGLE_SQUARE_STEPS) as i64
                } 
            } else {
                let offset = channel - KNIGHT_MOVE_START_IDX;
                match offset {
                    0 | 2 => -1,
                    1 | 3 => -2,
                    4 | 6 => 1,
                    5 | 7 => 2,
                    _ => unreachable!()
                }
            };

        if self.player == Color::Black {
            rank_diff *= -1;
            row = 7 - row;
        }

        let source_square = Square::make_square(
            Rank::from_index(row),
            File::from_index(col)
        );
        let dest_square = Square::make_square(
            Rank::from_index((row as i64 + rank_diff) as usize),
            File::from_index((col as i64 + file_diff) as usize)
        );

        ChessMove::new(source_square, dest_square, promotion)
    }
}

impl super::Policy for Policy {
    type Action = ChessMove;

    fn get_prob(&self, action: &Self::Action) -> f32 {
        let source_square = action.get_source();
        let mut row = source_square.get_rank().to_index();
        if self.player == Color::Black {
            row = 7 - row;
        }
        self.probs[[self.get_channel(action), row, source_square.get_file().to_index()]]
    }

    fn set_prob(&mut self, action: &Self::Action, prob: f32) {
        let source_square = action.get_source();
        let mut row = source_square.get_rank().to_index();
        if self.player == Color::Black {
            row = 7 - row;
        }
        let channel = self.get_channel(action);
        self.probs[[channel, row, source_square.get_file().to_index()]] = prob;
    }

    fn normalize(&mut self) {
        self.probs /= self.probs.sum()
    }

    fn get_flat_ndarray(&self) -> ndarray::Array1<f32> {
        self.probs.clone().into_shape((73 * 8 * 8,)).unwrap()
    }

    fn sample(&self, rng: &mut ThreadRng, temperature: f32) -> Self::Action {
        // Higher temperature => squishes probabilities together => encourages more exploration
        let temperature_action_probs = self.probs.mapv(|x| x.powf(temperature));
        let dist = WeightedIndex::new(temperature_action_probs.into_shape(73 * 8 * 8,).unwrap()).unwrap();
        let idx = dist.sample(rng);

        self.get_action(idx / 64, (idx % 64) / 8, idx % 8)
    }

    fn get_best_action(&self) -> Self::Action {
        let best_action_idx = self.probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();

        self.get_action(
            best_action_idx / 64, 
            (best_action_idx % 64) / 8, 
            best_action_idx % 8
        )
    }
}