mod tictactoe;

fn main() {
    let mut state: tictactoe::State = Default::default();
    let mut action = tictactoe::Action{row: 0, col: 0};
    state = state.get_next_state(&action).unwrap();
    println!("{state}");
    
    action.row = 2;
    action.col = 1;
    state = state.get_next_state(&action).unwrap();
    println!("{state}");
}
