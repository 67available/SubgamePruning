use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
};

use crate::tree::tree_struct::{self, *};

pub fn from_json(
    file_name: &str,
) -> (
    Vec<tree_struct::PubNode>,
    HashMap<usize, Vec<tree_struct::InfoSet>>,
    Vec<tree_struct::GameState>,
) {
    let data = fs::read_to_string(file_name).unwrap();

    let res: (Vec<PubNode>, HashMap<usize, Vec<InfoSet>>, Vec<GameState>) =
        serde_json::from_str(&data).unwrap();

    return res;
}

pub fn to_json(
    trees: (Vec<PubNode>, HashMap<usize, Vec<InfoSet>>, Vec<GameState>),
    file_name: &str,
) {
    // create_dir_all(dir_name).unwrap();
    let mut Data_Write_To_Output_In = BufWriter::new(File::create(file_name).unwrap());
    write!(
        &mut Data_Write_To_Output_In,
        "{}",
        serde_json::to_string(&trees).unwrap()
    )
    .unwrap();
}

#[test]
fn test_load_json() {
    let trees = from_json("tree_json/kuhn_poker/trees_kuhn_poker.txt");
    to_json(trees, "tree_test_to_json.txt");
}
