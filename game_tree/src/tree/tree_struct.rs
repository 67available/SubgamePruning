use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};

#[derive(Debug, Serialize, Deserialize)]
pub struct PubNode {
    pub infosets: HashMap<usize, Vec<usize>>,
    pub gamestates: Vec<usize>,
    pub father: Option<usize>,
    pub children: Vec<usize>,
    pub end_of_subtree: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InfoSet {
    pub player: i32, // -1 for chance
    pub key: String,
    pub index: usize,
    pub fahter: Option<usize>,
    pub children: Vec<usize>,
    pub i2state: Vec<usize>,
    pub pubnode: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct GameState {
    pub key: String,
    pub index: usize,
    pub father: Option<usize>,
    pub father_action: Option<usize>, // The index of the action taken to reach this node
    pub children: Vec<usize>,
    pub state2i: HashMap<usize, usize>,
    pub pubnode: usize,
    pub payoff: Option<Vec<f64>>, // Payoffs for each player; None if not a terminal node
    pub node_type: String,        // Node type: Chance / Play / Terminal (C / P / T)
    pub chance: Option<Vec<f64>>, // If a Chance node, specifies the action probabilities
    pub player: Option<usize>,    // If a Play node, indicates the acting player (0 / 1 / None)
}

#[derive(Debug, Clone)]
pub struct UtilityGameState {
    pub utitlity: Vec<f64>, // u_i(h)
    pub prob_r2h: Vec<f64>, // pi(h|r)
    pub prob_r: Vec<f64>,   // pi_i(r)
    pub root: usize,        // root game state index
}

impl UtilityGameState {
    pub fn new(gs: &GameState, index: usize) -> Self {
        if gs.children.len() == 0 {
            UtilityGameState {
                utitlity: <std::option::Option<Vec<f64>> as Clone>::clone(&gs.payoff).unwrap(),
                prob_r2h: vec![1.0; gs.state2i.len() + 1],
                prob_r: vec![1.0; gs.state2i.len() + 1],
                root: index,
            }
        } else {
            UtilityGameState {
                utitlity: vec![0.0; gs.state2i.len()],
                prob_r2h: vec![1.0; gs.state2i.len() + 1],
                prob_r: vec![1.0; gs.state2i.len() + 1],
                root: index,
            }
        }
    }
}

impl fmt::Display for UtilityGameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UtilityGameState {{ utitlity: [")?;
        for (i, val) in self.utitlity.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.3}", val)?;
        }
        write!(f, "], prob_r2h: [")?;
        for (i, val) in self.prob_r2h.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.3}", val)?;
        }
        write!(f, "], prob_r: [")?;
        for (i, val) in self.prob_r.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.3}", val)?;
        }
        write!(f, "], root: {} }}", self.root)
    }
}

pub fn build_ugs(game_tree: &Vec<GameState>) -> Vec<UtilityGameState> {
    // Initialize the utility game state vector based on the game state vector
    let mut ugs = vec![];
    game_tree
        .iter()
        .enumerate()
        .for_each(|(index, x)| ugs.push(UtilityGameState::new(x, index)));
    ugs
}
