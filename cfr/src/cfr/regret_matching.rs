use super::cfr_util::{check_nan, linear_combine};
use game_tree::tree::tree_struct::InfoSet;
use std::fmt;

#[derive(Debug, Clone)]
pub struct RMCell {
    pub accu_regret: Vec<f64>, // accumulated regret
    pub avr_prob: Vec<f64>,    // average strategy
    pub weight: f64,           // accumulated weight of average strategy
    pub max_regret: f64,       // max immediate regret
}

impl RMCellTrait for RMCell {
    const ALGO_NAME: &'static str = "VCFR";
    fn new(infoset: &InfoSet) -> Self {
        let num_actions = infoset.children.len();
        RMCell {
            accu_regret: vec![0.0; num_actions],
            avr_prob: vec![1.0 / num_actions as f64; num_actions],
            weight: 0.0,
            max_regret: f64::MAX,
        }
    }
    fn get_cur(&self) -> Vec<f64> {
        let s = self.accu_regret.iter().filter(|&&x| x > 0.0).sum::<f64>();
        let res;
        if s > 1e-10 {
            res = self
                .accu_regret
                .iter()
                .map(|&x| if x > 0.0 { x / s } else { 0.0 })
                .collect::<Vec<f64>>();
        } else {
            res = vec![1.0 / self.accu_regret.len() as f64; self.accu_regret.len()];
        }
        check_nan(&res);
        res
    }
    fn get_avr(&self) -> Vec<f64> {
        check_nan(&self.avr_prob);
        self.avr_prob.clone()
    }
    fn get_max_regret(&self) -> f64 {
        self.max_regret
    }
    fn set_max_regret(&mut self, r: f64) {
        self.max_regret = r;
    }
    fn get_type() -> String {
        "VCFR".to_string()
    }
    fn get_times(&self) -> usize {
        todo!()
    }
    fn rm(&mut self, regret: &Vec<f64>) {
        self.accu_regret
            .iter_mut()
            .enumerate()
            .for_each(|(action_index, r)| *r += regret[action_index]);
    }
    fn avr_update(&mut self, weight: f64, times: usize) {
        if self.weight + weight > 1e-10 {
            let prob = self.get_cur();
            linear_combine(&mut self.avr_prob, &prob, self.weight, weight);
        }
        self.weight = self.weight + weight;
    }

    fn get_accu_regret(&self) -> Vec<f64> {
        self.accu_regret.clone()
    }
}

pub fn build_rmc<T: RMCellTrait>(infoset_tree: &Vec<InfoSet>) -> Vec<T> {
    // build RMC for infoset Tree
    let mut rmc = vec![];
    infoset_tree.iter().for_each(|x| rmc.push(T::new(x)));
    rmc
}

impl fmt::Display for RMCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RMCell {{ accu_regret: {:?}, avr_prob: {:?}, weight: {:.3}, max_regret: {:.3} }}",
            self.accu_regret, self.avr_prob, self.weight, self.max_regret
        )
    }
}

pub trait RMCellTrait {
    const ALGO_NAME: &'static str;
    fn new(x: &InfoSet) -> Self;
    fn get_type() -> String;
    fn get_avr(&self) -> Vec<f64>;
    fn get_cur(&self) -> Vec<f64>;
    fn rm(&mut self, regret: &Vec<f64>);
    fn avr_update(&mut self, weight: f64, times: usize);
    fn get_max_regret(&self) -> f64;
    fn set_max_regret(&mut self, r: f64);
    fn get_times(&self) -> usize;
    fn get_accu_regret(&self) -> Vec<f64>;
}
