use crate::config::Config;
use std::collections::HashMap;

use super::{
    cfr_util::{check_nan, compute_prob_r_kldist, hashmap_to_sorted_vec},
    regret_matching::RMCell,
    regret_matching_plus::RMPlusCell,
};

pub trait PruneRMCell {
    fn prune_constraint_check(&self, regret: &Vec<f64>, config: &Config) -> bool;
}

impl PruneRMCell for RMCell {
    fn prune_constraint_check(&self, regret: &Vec<f64>, config: &Config) -> bool {
        let res = self
            .accu_regret
            .iter()
            .enumerate()
            .map(|(action_index, &x)| {
                if x > 0.0 && regret[action_index].abs() <= 1e-10 {
                    true
                } else if x <= 0.0 && regret[action_index] <= -x + 1e-10 {
                    true
                } else {
                    false
                }
            })
            .collect::<Vec<bool>>();
        if res.iter().all(|&x| x) {
            return true;
        } else {
            return false;
        }
    }
}

impl PruneRMCell for RMPlusCell {
    fn prune_constraint_check(&self, regret: &Vec<f64>, config: &Config) -> bool {
        let res = self
            .father
            .accu_regret
            .iter()
            .enumerate()
            .map(|(action_index, &x)| {
                if x > 0.0 && regret[action_index].abs() <= 1e-10 {
                    true
                } else if x <= 0.0 && regret[action_index] <= -x + 1e-10 {
                    true
                } else {
                    false
                }
            })
            .collect::<Vec<bool>>();
        if res.iter().all(|&x| x) {
            return true;
        } else {
            return false;
        }
    }
}

#[derive(Clone)]
pub struct SubgamePruneCell {
    pub prob_r_history: HashMap<usize, Vec<f64>>, // key: gamestate_index, value: accmulated historical reach probability
    pub prob_r_history4avr_update: HashMap<usize, Vec<f64>>, // Patch for CFR+
    pub max_regret_in_subgame: f64,               // δ-regret strategy
    pub prob_r_before_prune: Vec<Vec<f64>>, // reach probabiltiy at the moment the root was pruned
    pub prune_times: usize,                 // Patch for CFR+
    pub last_update_times: usize,           // Patch for CFR+
}

impl SubgamePruneCell {
    pub fn new(
        prob_r_before_prune: &HashMap<usize, Vec<f64>>,
        max_regret_in_subgame: f64,
        current_iteration: usize,
    ) -> Self {
        SubgamePruneCell {
            prob_r_history: prob_r_before_prune.clone(),
            prob_r_history4avr_update: prob_r_before_prune.clone(),
            max_regret_in_subgame,
            prob_r_before_prune: hashmap_to_sorted_vec(prob_r_before_prune),
            prune_times: 0,
            last_update_times: current_iteration,
        }
    }
    pub fn check_exempt_condition(
        &self,
        prob_r: &HashMap<usize, Vec<f64>>,
        config: &Config,
    ) -> bool {
        /* check-free condition verification */
        if self.max_regret_in_subgame < 1e-10 + config.subgame_prune.2 {
            // First, normalize the probabilities, as scaling the reach probability does not affect strategy choices within the subgame
            // Then, compute the distance between the two vectors
            let prob_r = hashmap_to_sorted_vec(prob_r);
            if compute_prob_r_kldist(&prob_r, &self.prob_r_before_prune)
                < 1e-8 + config.subgame_prune.3
            {
                true
            } else {
                false
            }
        } else {
            false
        }
    }
    pub fn update_prob_r_history(&mut self, mut prob_r: HashMap<usize, Vec<f64>>, cfrplus: bool) {
        /* accumulated reach probability for the root of the pruned subgame for the sake of RM compensation and Pruning Constraint Checking */
        self.prob_r_history
            .iter_mut()
            .for_each(|(gamestate_index, accu_prob_r)| {
                accu_prob_r
                    .iter_mut()
                    .zip(prob_r.get(gamestate_index).unwrap().iter())
                    .for_each(|(a, b)| *a += b);
                check_nan(&accu_prob_r);
            });
        if cfrplus {
            // If it is CFR+, multiply the incoming prob_r by the weight 'times'
            let weight = (self.last_update_times + 1 + self.prune_times);
            prob_r.iter_mut().for_each(|(gamestate_index, y)| {
                y.iter_mut()
                    .for_each(|x| *x *= weight as f64 / (self.last_update_times + 1) as f64)
            });
            self.prob_r_history4avr_update
                .iter_mut()
                .for_each(|(gamestate_index, accu_prob_r)| {
                    accu_prob_r
                        .iter_mut()
                        .zip(prob_r.get(gamestate_index).unwrap().iter())
                        .for_each(|(a, b)| *a += b);
                    check_nan(&accu_prob_r);
                });
        }
        self.prune_times += 1;
    }
}
