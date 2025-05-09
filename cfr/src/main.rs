use cfr::{
    agents::{RMAgent, StaticTrees, UgsAgent},
    cfr_iteration_v2::cfr_iteration,
    regret_matching::RMCell,
    regret_matching_plus::RMPlusCell,
};
use clap::{arg, value_parser, Arg, Command};
use config::Config;
use game_tree::load_json;
use std::collections::HashMap;
pub mod cfr;
pub mod config;

mod test;

const GAMES: [&str; 6] = [
    "kuhn_poker",
    "tiny_hanabi",
    "tiny_bridge_2p",
    "leduc_poker",
    "liars_dice",
    "my_leduc5",
];

fn main() {
    let matches = Command::new("SUBGAMEPRUNING")
        // Application configuration
        .about("This the intro of the cli application")
        // Application args
        .arg(arg!([MAIN]).help("fill in 'main'"))
        .arg(
            Arg::new("Algo")
                .short('a')
                .long("algo")
                .help("CFR/CFR+")
                .value_parser(value_parser!(String))
                .default_value("CFR"),
        )
        .arg(
            Arg::new("game_env")
                .short('g')
                .long("game")
                .help("0~5")
                .value_parser(value_parser!(usize))
                .default_value("0"),
        )
        .arg(
            Arg::new("iterations")
                .short('i')
                .long("iter")
                .value_parser(value_parser!(usize))
                .default_value("100"),
        )
        .arg(
            Arg::new("subgame_pruning")
                .short('s')
                .long("sp")
                .help("Enable Subgame Pruning")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("dir")
                .short('d')
                .long("dir")
                .help("output dir")
                .value_parser(value_parser!(String))
                .default_value("./results"),
        )
        .arg(
            Arg::new("num_thread")
                .short('t')
                .long("thread")
                .help("num of thread")
                .value_parser(value_parser!(usize))
                .default_value("0"),
        )
        .get_matches();
    // Read and parse command args
    let algo = matches.get_one::<String>("Algo").unwrap();
    let iterations = *matches.get_one::<usize>("iterations").unwrap();
    let subgame_pruning = *matches.get_one::<bool>("subgame_pruning").unwrap_or(&false);
    let game_index = *matches.get_one::<usize>("game_env").unwrap();
    let dir = matches.get_one::<String>("dir").unwrap().to_string();
    let num_thread = *matches.get_one::<usize>("num_thread").unwrap();

    if num_thread > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_thread)
            .build_global()
            .unwrap();
    }

    // load tree
    let game_name = GAMES[game_index];
    println!("game env {}", game_name);
    let trees: (
        Vec<game_tree::tree::tree_struct::PubNode>,
        HashMap<usize, Vec<game_tree::tree::tree_struct::InfoSet>>,
        Vec<game_tree::tree::tree_struct::GameState>,
    ) = load_json::from_json(&format!(
        "game_tree/tree_json/{}/trees_{}.txt",
        game_name, game_name
    ));
    println!(
        "size of game state tree {}, size of public tree {}",
        trees.2.len(),
        trees.0.len(),
    );
    let mut ugs = UgsAgent::new(&trees);
    let mut config;
    if algo == "CFR+" {
        let mut rmc: RMAgent<RMPlusCell> = RMAgent::new(&trees);
        let static_trees = StaticTrees::new(trees);
        if !subgame_pruning {
            config = Config::new(
                game_name.to_string(),
                rmc.algo_name.clone(),
                false,
                false,
                dir,
            );
        } else {
            config = Config::new_subgame_prune_with_check_free(
                game_name.to_string(),
                rmc.algo_name.clone(),
                true,
                true,
                dir,
            );
        }
        println!("config {}", config.to_string());
        let mut record_infosets = vec![vec![]; 2];
        for i in 0..static_trees.trees.0.len() {
            if static_trees.get_pubnode_type(i) == 'P'.to_string() {
                let p = static_trees.get_pubnode_player(i);
                for infoset_index in static_trees.trees.0[i].infosets.get(&p).unwrap() {
                    record_infosets[p].push(*infoset_index);
                    if record_infosets[p].len() >= 3 {
                        break;
                    }
                }
            }
            if record_infosets[0].len() >= 3 && record_infosets[1].len() >= 3 {
                break;
            }
        }
        cfr_iteration(
            &static_trees,
            &mut ugs,
            &mut rmc,
            iterations as usize + 1,
            &mut config,
            record_infosets,
        );
    } else {
        let mut rmc: RMAgent<RMCell> = RMAgent::new(&trees);
        let static_trees = StaticTrees::new(trees);
        if !subgame_pruning {
            config = Config::new(
                game_name.to_string(),
                rmc.algo_name.clone(),
                false,
                false,
                dir,
            );
        } else {
            config = Config::new_subgame_prune_with_check_free(
                game_name.to_string(),
                rmc.algo_name.clone(),
                true,
                true,
                dir,
            );
        }
        println!("config {}", config.to_string());
        let mut record_infosets = vec![vec![]; 2]; // info-sets whose accumulated regret will be recorded and plot
        for i in 0..static_trees.trees.0.len() {
            if static_trees.get_pubnode_type(i) == 'P'.to_string() {
                let p = static_trees.get_pubnode_player(i);
                for infoset_index in static_trees.trees.0[i].infosets.get(&p).unwrap() {
                    record_infosets[p].push(*infoset_index);
                    if record_infosets[p].len() >= 3 {
                        break;
                    }
                }
            }
            if record_infosets[0].len() >= 3 && record_infosets[1].len() >= 3 {
                break;
            }
        }
        cfr_iteration(
            &static_trees,
            &mut ugs,
            &mut rmc,
            iterations as usize + 1,
            &mut config,
            record_infosets,
        );
    }
}
