use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{self, Display},
    fs::{self, File, create_dir_all},
    io::{BufWriter, Write},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct PubNode {
    pub infosets: Vec<Vec<String>>,
    pub histories: Vec<usize>,
    pub father: Option<usize>,
    pub children: Vec<usize>,
    pub node_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InfoSet {
    pub player: i32,
    pub key: String,
    pub index: usize,
    pub father: Option<usize>,
    pub children: Vec<usize>,
    pub i2history: Vec<usize>,
    pub pubnode: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct History {
    pub key: String,
    pub index: usize,
    pub current_player: i32, // 当前决策玩家
    pub father: Option<usize>,
    pub father_action: Option<String>, // 做了哪个动作到达这里的
    pub payoff: Option<Vec<f64>>,      // 每个玩家的收益 None/dict
    pub chance: Option<Vec<f64>>,      // 如果是Chance节点的话需要知道动作概率 list
    pub node_type: String,             // 节点类型Chance/Play/Terminal -> CPT
    pub children: Vec<usize>,
    pub history2i: Vec<String>,
    pub history2i_idx: Vec<usize>,
    pub pubnode: usize,
}

pub fn from_json(
    file_name: &str,
) -> (
    Vec<PubNode>,
    Vec<Vec<InfoSet>>,
    Vec<HashMap<String, InfoSet>>,
    Vec<History>,
) {
    // 读取 JSON 文件
    let data = fs::read_to_string(file_name).unwrap();
    // 解析 JSON 数据
    let trees: (
        Vec<PubNode>,
        Vec<Vec<InfoSet>>,
        Vec<HashMap<String, InfoSet>>,
        Vec<History>,
    ) = serde_json::from_str(&data).unwrap();
    return trees;
}

#[test]
fn test_load_json() {
    let games: [&str; 6] = [
        "kuhn_poker",           // ✔ ✔ 第二个勾代表通过cfr测试
        "tiny_hanabi",          // ✔ ⬛️ 在第三代就找到了纳什均衡，之后有空检查一下
        "liars_dice", // ✔ 存在问题？死循环？没问题，只是比较慢 // 500代内没有发生prune 好像根本不会发生prune
        "first_sealed_auction", // ✔ ✔
        "leduc_poker", // ✔
        "tiny_bridge_2p", // ✔
    ];
    for game_name in games {
        println!("loading {}", game_name);
        let trees: (
            Vec<PubNode>,
            Vec<Vec<InfoSet>>,
            Vec<HashMap<String, InfoSet>>,
            Vec<History>,
        ) = from_json(&format!("tree/{}/trees_{}_0923.txt", game_name, game_name));
        println!(
            "game_name: {} \n pubtree size: {} | infoset_tree size: {:?} | game_state_tree size {}",
            game_name,
            trees.0.len(),
            trees.1.iter().map(|x| x.len()).collect::<Vec<_>>(),
            trees.3.len()
        );

        let pubtree_len = trees.0.len();
        let infoset_sizes: Vec<usize> = trees.1.iter().map(|x| x.len()).collect();
        let gamestate_len = trees.3.len();

        match game_name {
            "kuhn_poker" => {
                assert_eq!(pubtree_len, 11, "kuhn poker: pubtree size mismatch");
                assert_eq!(
                    infoset_sizes,
                    vec![31, 29],
                    "kuhn poker: infoset sizes mismatch"
                );
                assert_eq!(gamestate_len, 58, "kuhn poker: game state size mismatch");
            }
            "tiny_hanabi" => {
                assert_eq!(pubtree_len, 15, "tiny hanabi: pubtree size mismatch");
                assert_eq!(
                    infoset_sizes,
                    vec![29, 28],
                    "tiny hanabi: infoset sizes mismatch"
                );
                assert_eq!(gamestate_len, 55, "tiny hanabi: game state size mismatch");
            }
            "liars_dice" => {
                assert_eq!(pubtree_len, 8193, "liars dice: pubtree size mismatch");
                assert_eq!(
                    infoset_sizes,
                    vec![49153, 49148],
                    "liars dice: infoset sizes mismatch"
                );
                assert_eq!(
                    gamestate_len, 294_883,
                    "liars dice: game state size mismatch"
                );
            }
            "first_sealed_auction" => {
                assert_eq!(
                    pubtree_len, 6,
                    "first sealed auction: pubtree size mismatch"
                );
                assert_eq!(
                    infoset_sizes,
                    vec![186, 132],
                    "first sealed auction: infoset sizes mismatch"
                );
                assert_eq!(
                    gamestate_len, 7_096,
                    "first sealed auction: game state size mismatch"
                );
            }
            "leduc_poker" => {
                assert_eq!(pubtree_len, 467, "leduc poker: pubtree size mismatch");
                assert_eq!(
                    infoset_sizes,
                    vec![2347, 2342],
                    "leduc poker: infoset sizes mismatch"
                );
                assert_eq!(
                    gamestate_len, 9_457,
                    "leduc poker: game state size mismatch"
                );
            }
            "tiny_bridge_2p" => {
                assert_eq!(pubtree_len, 257, "tiny bridge 2p: pubtree size mismatch");
                assert_eq!(
                    infoset_sizes,
                    vec![7169, 7142],
                    "tiny bridge 2p: infoset sizes mismatch"
                );
                assert_eq!(
                    gamestate_len, 107_129,
                    "tiny bridge 2p: game state size mismatch"
                );
            }
            "tic_tac_toe" => {
                assert_eq!(pubtree_len, 549_946, "tic tac toe: pubtree size mismatch");
                assert_eq!(
                    infoset_sizes,
                    vec![549_946, 549_946],
                    "tic tac toe: infoset sizes mismatch"
                );
                assert_eq!(
                    gamestate_len, 549_946,
                    "tic tac toe: game state size mismatch"
                );
            }
            "my_leduc5" => {
                assert_eq!(pubtree_len, 67_162, "my leduc5: pubtree size mismatch");
                assert_eq!(
                    infoset_sizes,
                    vec![335_989, 335_989],
                    "my leduc5: infoset sizes mismatch"
                );
                assert_eq!(
                    gamestate_len, 1_345_051,
                    "my leduc5: game state size mismatch"
                );
            }
            other => panic!("Unknown game_name: {other}"),
        }
    }
}

// cargo test test_load_json -- --nocapture
