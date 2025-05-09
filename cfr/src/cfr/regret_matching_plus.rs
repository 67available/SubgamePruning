use game_tree::tree::tree_struct::InfoSet;
use std::fmt;

use super::{
    cfr_util::{check_nan, linear_combine},
    regret_matching::{RMCell, RMCellTrait},
};

#[derive(Debug, Clone)]
pub struct RMPlusCell {
    pub father: RMCell,
    times: usize,
}

impl RMCellTrait for RMPlusCell {
    fn new(infoset: &InfoSet) -> Self {
        RMPlusCell {
            father: RMCell::new(infoset),
            times: 1,
        }
    }
    fn rm(&mut self, regret: &Vec<f64>) {
        self.father
            .accu_regret
            .iter_mut()
            .enumerate()
            .for_each(|(action_index, r)| {
                if *r + regret[action_index] > 0.0 {
                    *r = *r + regret[action_index]
                } else {
                    *r = 0.0
                }
            });
    }
    fn avr_update(&mut self, weight: f64, times: usize) {
        let weight = self.times as f64 * weight;
        let cur = self.father.get_cur();
        if self.father.weight + weight > 1e-10 {
            linear_combine(&mut self.father.avr_prob, &cur, self.father.weight, weight);
        }
        self.father.weight = self.father.weight + weight;
        self.times += times;
        check_nan(&self.father.avr_prob);
    }
    fn get_avr(&self) -> Vec<f64> {
        self.father.get_avr()
    }
    fn get_cur(&self) -> Vec<f64> {
        self.father.get_cur()
    }
    fn get_max_regret(&self) -> f64 {
        self.father.get_max_regret()
    }
    fn set_max_regret(&mut self, r: f64) {
        self.father.set_max_regret(r);
    }
    fn get_type() -> String {
        "CFR+".to_string()
    }
    fn get_times(&self) -> usize {
        self.times
    }

    const ALGO_NAME: &'static str = "CFR+";

    fn get_accu_regret(&self) -> Vec<f64> {
        self.father.get_accu_regret()
    }
}

pub fn build_rmplsc(infoset_tree: &Vec<InfoSet>) -> Vec<RMPlusCell> {
    // build RMC for infoset Tree
    let mut rmc = vec![];
    infoset_tree
        .iter()
        .for_each(|x| rmc.push(RMPlusCell::new(x)));
    rmc
}

impl fmt::Display for RMPlusCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RMPlusCell {{ father: {}, times: {} }}",
            self.father, self.times
        )
    }
}
