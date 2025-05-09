use super::agents::SubgamePruneAgent;
use super::cfr_util::max_index4vec;
use super::{
    agents::{
        get_br_strategy4pubnode, get_cv4infoset, get_strategy4pubnode, RMAgent, StaticTrees,
        UgsAgent,
    },
    cfr_util, regret_matching,
    subgame_prune_v2::PruneRMCell,
};
use crate::{cfr::cfr_util::flatten_outer_vec, config};
use cfr_util::transpose;
use config::Config;
use csv_tool::tool::write_csv;
use game_tree::tree::tree_struct::PubNode;
use plot::plot_loss_curves;

use rayon::prelude::*;
use regret_matching::RMCellTrait;
use std::collections::{HashMap, HashSet};

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use tqdm::tqdm;

fn compute_br4p<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    br_player: usize,
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    rmc: &RMAgent<T>,
) {
    let shared_ugs = Arc::new(Mutex::new(ugs));
    let mut queue = VecDeque::new();
    let mut rev_queue = VecDeque::new();
    queue.push_back(0);

    while !queue.is_empty() {
        rev_queue.push_back(queue.clone());
        let level_size = queue.len();
        let mut level_nodes = Vec::with_capacity(level_size);
        for _ in 0..level_size {
            let index = queue.pop_front().unwrap();
            let node = &static_trees.trees.0[index];
            level_nodes.push(index);
            for &child_index in &node.children {
                queue.push_back(child_index);
            }
        }
        level_nodes.par_iter().for_each(|pubnode_index| {
            let (strategy, game_state_cluster) =
                get_strategy4pubnode(&static_trees, rmc, *pubnode_index, false);
            let mut ugs_lock = shared_ugs.lock().unwrap();
            for (s, game_states) in strategy.iter().zip(game_state_cluster.iter()) {
                for game_state_index in game_states.iter() {
                    ugs_lock.update_children_rp(*game_state_index, s);
                }
            }
        });
    }
    while rev_queue.len() > 0 {
        let mut queue = rev_queue.pop_back().unwrap();
        let level_size = queue.len();
        let mut level_nodes = Vec::with_capacity(level_size);
        for _ in 0..level_size {
            let index = queue.pop_front().unwrap();
            level_nodes.push(index);
        }
        level_nodes.par_iter().for_each(|&pubnode_index| {
            let mut ugs_lock = shared_ugs.lock().unwrap();
            let current_player = static_trees.get_pubnode_player(pubnode_index);
            // get BR strategy
            let (strategy, game_state_cluster) = if current_player == br_player {
                get_br_strategy4pubnode(&static_trees, &ugs_lock, pubnode_index)
            } else {
                get_strategy4pubnode(&static_trees, &rmc, pubnode_index, false)
            };
            for (s, game_states) in strategy.iter().zip(game_state_cluster.iter()) {
                for game_state_index in game_states.iter() {
                    ugs_lock.update_self_utitlity(*game_state_index, s);
                }
            }
        });
    }
}

fn compute_expect_return<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    rmc: &RMAgent<T>,
    spc: &SubgamePruneAgent,
    config: &mut Config,
) -> Vec<f64> {
    cfr_iteration_stage1(static_trees, ugs, rmc, spc, false, config);
    ugs.get_utitliy_vec(0)
}

fn compute_br_return<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    rmc: &RMAgent<T>,
    config: &mut Config,
) -> Vec<f64> {
    /*
    // To compute the best response (BR) strategy, stage 1 must first be executed using the average strategy
    // to obtain an accurate realization plan. During the utility update phase, the BR strategy should be used for the update.
     */
    let mut br = vec![];
    compute_br4p(0, &static_trees, ugs, &rmc);
    br.push(ugs.get_utitliy_vec(0)[0]);
    compute_br4p(1, &static_trees, ugs, &rmc);
    br.push(ugs.get_utitliy_vec(0)[1]);
    br
}

fn reset_gamestate_root(
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    compensate_roots: Vec<usize>,
    config: &mut Config,
) {
    for pubnode_index in compensate_roots.iter() {
        static_trees.trees.0[*pubnode_index]
            .gamestates
            .iter()
            .for_each(|&gamestate_index| ugs.reset_gamestate_from_root(gamestate_index));
    }
}

fn set_probr2h_and_root<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    rmc: &RMAgent<T>,
    spc: &SubgamePruneAgent,
    first_pruned_roots: Vec<usize>,
    config: &mut Config,
) {
    /*
    // For all under-pruned subtrees:
    // modify the root properties of the pruned subtrees and
    // the conditional probabilities from the root to the nodes within the subtree
    // (only needs to be done once)
     */
    for pubnode_index in first_pruned_roots.iter() {
        static_trees.trees.0[*pubnode_index]
            .gamestates
            .iter()
            .for_each(|&gamestate_index| ugs.update_gamestate_from_root(gamestate_index));
    }
    let shared_ugs = Arc::new(Mutex::new(ugs));
    let mut queue = VecDeque::new();
    let mut rev_queue = VecDeque::new();
    first_pruned_roots.iter().for_each(|x| queue.push_back(*x));
    while !queue.is_empty() {
        rev_queue.push_back(queue.clone());
        let level_size = queue.len();
        let mut level_nodes = Vec::with_capacity(level_size);
        for _ in 0..level_size {
            let index = queue.pop_front().unwrap();
            let node = &static_trees.trees.0[index];
            level_nodes.push(index);
            for &child_index in &node.children {
                queue.push_back(child_index);
            }
        }
        level_nodes.par_iter().for_each(|pubnode_index| {
            let (strategy, game_state_cluster) =
                get_strategy4pubnode(&static_trees, &rmc, *pubnode_index, false);
            let mut ugs_lock = shared_ugs.lock().unwrap();
            for (s, game_states) in strategy.iter().zip(game_state_cluster.iter()) {
                for game_state_index in game_states.iter() {
                    ugs_lock.update_children_rp(*game_state_index, s);
                }
            }
        });
    }
}

fn cfr_iteration_stage1<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    rmc: &RMAgent<T>,
    spc: &SubgamePruneAgent,
    avr: bool, // Set to true to compute the overall expected utility under the average strategy, otherwise current strategy with be retrieved
    config: &mut Config,
) {
    /* Reach Probability and Utility Update Step
    // Perform a forward breadth-first traversal (BFS) of the public tree to extract strategies
    // and compute realization plans. For each public tree node, traverse all
    // associated game states and update the realization plans of their child nodes.
     */
    let shared_ugs = Arc::new(Mutex::new(ugs));
    let mut queue = VecDeque::new();
    let mut rev_queue = VecDeque::new(); //for reversed BFS
    queue.push_back(0);
    while !queue.is_empty() {
        rev_queue.push_back(queue.clone());
        let level_size = queue.len();
        let mut level_nodes = Vec::with_capacity(level_size);
        for _ in 0..level_size {
            let index = queue.pop_front().unwrap();
            let node = &static_trees.trees.0[index];
            level_nodes.push(index);
            if !spc.pruned_roots_contain(index) {
                for &child_index in &node.children {
                    queue.push_back(child_index);
                }
            } // stage 1 is omitted for all non-root pruned nodes
        }
        level_nodes.par_iter().for_each(|pubnode_index| {
            let (strategy, game_state_cluster) =
                get_strategy4pubnode(&static_trees, &rmc, *pubnode_index, avr); // strategy request
            let mut ugs_lock = shared_ugs.lock().unwrap();
            for (s, game_states) in strategy.iter().zip(game_state_cluster.iter()) {
                for game_state_index in game_states.iter() {
                    ugs_lock.update_children_rp(*game_state_index, s); // update reach probability
                }
            }
        });
    }
    // update utitliy of game states by reversed BFS
    while rev_queue.len() > 0 {
        let mut queue = rev_queue.pop_back().unwrap();
        let level_size = queue.len();
        let mut level_nodes = Vec::with_capacity(level_size);
        for _ in 0..level_size {
            let index = queue.pop_front().unwrap();
            level_nodes.push(index);
        }
        level_nodes.par_iter().for_each(|pubnode_index| {
            let (strategy, game_state_cluster) =
                get_strategy4pubnode(&static_trees, &rmc, *pubnode_index, avr);
            let mut ugs_lock = shared_ugs.lock().unwrap();
            for (s, game_states) in strategy.iter().zip(game_state_cluster.iter()) {
                for game_state_index in game_states.iter() {
                    ugs_lock.update_self_utitlity(*game_state_index, s);
                }
            }
        });
    }
}

fn cfr_iteration_stage2<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &UgsAgent,
    rmc: &mut RMAgent<T>,
    spc: &SubgamePruneAgent,
    config: &mut Config,
) {
    /* Average Strategy Update Step
    // Update the average strategy for each RMC:
    // for each RMC, obtain the player's own contribution to realization plan (reach probabiltiy), i.e., the weight of the current strategy),
    // and then invoke RMC's corresponding update method.
     */
    let shared_rmc = Arc::new(Mutex::new(rmc));
    static_trees
        .trees
        .0
        .par_iter()
        .enumerate()
        .filter(|(pubnode_index, _)| {
            !(spc.check_under_pruned(*pubnode_index).0
                || spc.check_under_compensated(*pubnode_index).0) // 既没有被pruned也没被compensate
        })
        .for_each(|(pubnode_index, pubnode)| {
            let p = static_trees.get_pubnode_player(pubnode_index);
            let node_type = static_trees.get_pubnode_type(pubnode_index);
            if node_type == "P".to_string() {
                for rmc_index in pubnode.infosets.get(&p).unwrap().iter() {
                    let game_state_index =
                        static_trees.trees.1.get(&p).unwrap()[*rmc_index].i2state[0];
                    let my_realization_plan =
                        ugs.get_realization_plan4p(game_state_index, p, false, None);
                    let mut rmc_lock = shared_rmc.lock().unwrap();
                    rmc_lock.update_avr_strategy(p, *rmc_index, my_realization_plan, 1);
                }
            }
        });
}

fn cfr_iteration_stage3<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &UgsAgent,
    rmc: &mut RMAgent<T>,
    spc: &SubgamePruneAgent,
    config: &mut Config,
) {
    /* Regret Matching Step */
    let shared_rmc = Arc::new(Mutex::new(rmc));
    let mut queue = VecDeque::new();
    let mut rev_queue = VecDeque::new();
    queue.push_back(0);
    while !queue.is_empty() {
        rev_queue.push_back(queue.clone());
        let level_size = queue.len();
        for _ in 0..level_size {
            let index = queue.pop_front().unwrap();
            let node = &static_trees.trees.0[index];
            if !(spc.pruned_roots_contain(index) || spc.check_under_compensated(index).0) {
                for &child_index in &node.children {
                    queue.push_back(child_index);
                }
            }
        }
    }
    while rev_queue.len() > 0 {
        let mut queue = rev_queue.pop_back().unwrap();
        let level_size = queue.len();
        let mut level_nodes = Vec::with_capacity(level_size);
        for _ in 0..level_size {
            let index = queue.pop_front().unwrap();
            level_nodes.push(index);
        }
        level_nodes
            .par_iter()
            .filter(|(&pubnode_index)| {
                !(spc.check_under_pruned(*pubnode_index).0
                    || spc.check_under_compensated(*pubnode_index).0) // Nodes that are neither pruned nor compensated in current iteration
            })
            .for_each(|&pubnode_index| {
                let actor = static_trees.get_pubnode_player(pubnode_index);
                let node_type = static_trees.get_pubnode_type(pubnode_index);
                let pubnode = &static_trees.trees.0[pubnode_index];
                if node_type == "P".to_string() {
                    for &rmc_index in pubnode.infosets.get(&actor).unwrap().iter() {
                        let cv_i = get_cv4infoset(static_trees, ugs, actor, rmc_index, None);
                        let cv_ia = static_trees.trees.1[&actor][rmc_index]
                            .children
                            .iter()
                            .map(|&x| get_cv4infoset(static_trees, ugs, actor, x, None))
                            .collect::<Vec<f64>>();
                        let regret = cv_ia.iter().map(|&x| x - cv_i).collect::<Vec<f64>>();
                        let mut rmc_lock = shared_rmc.lock().unwrap();
                        rmc_lock.update_accu_regret(actor, rmc_index, &regret);
                    }
                }
            });
    }
}

pub fn get_prob_r4pubnode(
    pubnode_index: usize,
    static_trees: &StaticTrees,
    ugs: &UgsAgent,
    history_prob_r: Option<&HashMap<usize, Vec<f64>>>,
) -> HashMap<usize, Vec<f64>> {
    let num_player = static_trees.num_player;
    let prob_r = static_trees.trees.0[pubnode_index]
        .gamestates
        .iter()
        .map(|x| {
            (
                *x,
                (0..num_player)
                    .into_iter()
                    .map(|p| ugs.get_realization_plan4p(*x, p, false, history_prob_r))
                    .collect::<Vec<f64>>(),
            )
        })
        .collect::<HashMap<usize, Vec<f64>>>();
    prob_r
}

fn check_free_subgame_checking<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &UgsAgent,
    rmc: &RMAgent<T>,
    spc: &mut SubgamePruneAgent,
    config: &mut Config,
) {
    let pruned_subgame = &spc.get_pruned_subgame();
    let mut res = HashSet::new();
    pruned_subgame.iter().for_each(|&pubnode_index| {
        let prob_r = get_prob_r4pubnode(pubnode_index, static_trees, ugs, None);
        if spc.check_check_free_node(pubnode_index, &prob_r, config) {
            res.insert(pubnode_index);
        }
    });
    spc.update_check_free_roots(res, static_trees, &config);
}

fn rm_compensation<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &UgsAgent,
    rmc: &mut RMAgent<T>,
    spc: &mut SubgamePruneAgent,
    config: &mut Config,
) {
    /* Regret Matching Compensation */
    let shared_rmc = Arc::new(Mutex::new(rmc));
    spc.get_compensation_roots() // Traverse all subtrees (roots) that require compensation
        .into_iter()
        .for_each(|pubnode_index| {
            let history_prob_r;
            if config.algo_name == "CFR+".to_string() {
                history_prob_r = spc.get_history_prob_r4pubnode(pubnode_index, true);
            } else {
                history_prob_r = spc.get_history_prob_r4pubnode(pubnode_index, false);
            }
            let mut queue = VecDeque::new();
            queue.push_back(pubnode_index);
            while !queue.is_empty() {
                let level_size = queue.len();
                let mut level_nodes = Vec::with_capacity(level_size);
                for _ in 0..level_size {
                    let index = queue.pop_front().unwrap();
                    let node = &static_trees.trees.0[index];
                    level_nodes.push(index);
                    for &child_index in &node.children {
                        queue.push_back(child_index);
                    }
                }
                level_nodes.par_iter().for_each(|&pubnode_index| {
                    let actor = static_trees.get_pubnode_player(pubnode_index);
                    let node_type = static_trees.get_pubnode_type(pubnode_index);
                    let pubnode = &static_trees.trees.0[pubnode_index];
                    let mut rmc_lock = shared_rmc.lock().unwrap();
                    if node_type == "P".to_string() {
                        for &rmc_index in pubnode.infosets.get(&actor).unwrap().iter() {
                            let game_state_index =
                                static_trees.trees.1.get(&actor).unwrap()[rmc_index].i2state[0];
                            let my_realization_plan = ugs.get_realization_plan4p(
                                game_state_index,
                                actor,
                                true,
                                Some(&history_prob_r),
                            );
                            rmc_lock.update_avr_strategy(
                                actor,
                                rmc_index,
                                my_realization_plan,
                                spc.get_times(pubnode_index),
                            );
                            let cv_i = get_cv4infoset(
                                static_trees,
                                ugs,
                                actor,
                                rmc_index,
                                Some(&history_prob_r),
                            );
                            let cv_ia = static_trees.trees.1[&actor][rmc_index]
                                .children
                                .iter()
                                .map(|&x| {
                                    get_cv4infoset(
                                        static_trees,
                                        ugs,
                                        actor,
                                        x,
                                        Some(&history_prob_r),
                                    )
                                })
                                .collect::<Vec<f64>>();
                            let regret = cv_ia.iter().map(|&x| x - cv_i).collect::<Vec<f64>>();
                            if config.algo_name == "VCFR" {
                                // accumulated regret compensation is omiited for CFR+
                                rmc_lock.update_accu_regret(actor, rmc_index, &regret);
                            }
                        }
                    }
                });
            }
        });
    for &pubnode_index in spc.get_compensation_roots().iter() {
        spc.remove(pubnode_index);
    }
}

fn prune_constraint_checking<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    rmc: &mut RMAgent<T>,
    spc: &mut SubgamePruneAgent,
    config: &mut Config,
) -> (Vec<usize>, Vec<usize>) {
    /* Pruning Constraint Checking
    // Need to return a list of pruned subgame roots.
    // Perform a reverse BFS traversal over each level (depth) of pubnodes:
    // (leaves with be autoly added into the candidates list)
    // 0. nodes whose num of votes if less than its num of child nodes will be kicked out from the candidates list
    // 1. first add the parent node of each node into the candidates list (with zero vote),
    // 2. then traverse the current level. For nodes that satisfy the pruning constraint,
    // remove their children from the pruned subgame root list and add the node itself (unless it is a terminal node).
    // Then, cast a vote for its parent node in the candidates list.
    // 3. move to upper level and start from step 0.
     */
    let mut queue = VecDeque::new();
    let candidates = Arc::new(Mutex::new(HashMap::new()));
    let pruned_roots = Arc::new(Mutex::new(HashSet::new()));
    let mut depth = static_trees.get_depth();
    let shared_rmc = Arc::new(Mutex::new(&mut *rmc));

    while depth > 0 {
        depth -= 1;
        for &i in static_trees.get_leaves_at_level(depth).iter() {
            queue.push_back(i); // initialize queue with leaves
        }
        let mut candidates_lock = candidates.lock().unwrap();
        for (&pubnode_index, &count) in candidates_lock.iter() {
            let pubnode: &PubNode = &static_trees.trees.0[pubnode_index];
            if pubnode.children.len() == count {
                queue.push_back(pubnode_index); // if all child nodes satisfy the pruning constraints, it will become a candidate
            }
        }
        candidates_lock.clear();
        let level_size = queue.len();
        let mut level_nodes = vec![];
        for _ in (0..level_size).into_iter() {
            let pubnode_index = queue.pop_front().unwrap();
            let pubnode = &static_trees.trees.0[pubnode_index];
            if let Some(father_index) = pubnode.father {
                candidates_lock.insert(father_index, 0);
            }
            level_nodes.push(pubnode_index);
        }
        drop(candidates_lock);
        level_nodes.par_iter().for_each(|&pubnode_index| {
            let actor = static_trees.get_pubnode_player(pubnode_index);
            let node_type = static_trees.get_pubnode_type(pubnode_index);
            let pubnode = &static_trees.trees.0[pubnode_index];
            let under_pruned = spc.check_under_pruned(pubnode_index);
            let mut history_prob_r = None;
            if under_pruned.0 {
                history_prob_r = Some(spc.get_history_prob_r4pubnode(under_pruned.1, false));
            }
            let mut pruned: bool = true;
            if pubnode.children.len() == 0 && false {
            } else if spc.whether_check_free(pubnode_index).0 { // check-free-node don't need to be checked
            } else {
                for p in 0..2 {
                    for &rmc_index in pubnode.infosets.get(&p).unwrap().iter() {
                        if node_type == "P".to_string() {
                            let cv_i = get_cv4infoset(
                                static_trees,
                                ugs,
                                p,
                                rmc_index,
                                history_prob_r.as_ref(),
                            );
                            let cv_ia = static_trees.trees.1[&p][rmc_index]
                                .children
                                .iter()
                                .map(|&x| {
                                    get_cv4infoset(static_trees, ugs, p, x, history_prob_r.as_ref())
                                })
                                .collect::<Vec<f64>>();
                            let regret = cv_ia.iter().map(|&x| x - cv_i).collect::<Vec<f64>>();
                            let max_regret = max_index4vec(&regret).0;
                            let mut rmc_lock = shared_rmc.lock().unwrap();
                            rmc_lock.set_max_regret4subgame(actor, rmc_index, max_regret);
                            if rmc_lock.prune_constraint_check(p, rmc_index, &regret, config)
                                == false
                            {
                                pruned = false;
                                break;
                            }
                        }
                    }
                }
            }
            if pruned {
                if let Some(father_index) = pubnode.father {
                    *candidates.lock().unwrap().get_mut(&father_index).unwrap() += 1;
                }
                for i in pubnode.children.iter() {
                    pruned_roots.lock().unwrap().remove(i);
                }
                if pubnode.children.len() > 0 {
                    // single leaf node would not be considered as a prunable subtree
                    pruned_roots.lock().unwrap().insert(pubnode_index);
                }
            }
        });
    }
    let pruned_roots = pruned_roots
        .lock()
        .unwrap()
        .iter()
        .map(|&x| x)
        .collect::<Vec<usize>>();
    spc.update_pruned_roots(pruned_roots, static_trees, ugs, rmc, config)
}

pub fn cfr_iteration<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
    static_trees: &StaticTrees,
    ugs: &mut UgsAgent,
    rmc: &mut RMAgent<T>,
    epoches: usize,
    config: &mut Config,
    record_accu_regret: Vec<Vec<usize>>,
) where
    T: Clone,
{
    let mut spc = SubgamePruneAgent::new();
    let mut expected_returns: Vec<Vec<f64>> = Vec::new();
    let mut bestres_returns: Vec<Vec<f64>> = Vec::new();
    let mut exploitablity: Vec<Vec<f64>> = Vec::new();
    let mut prune_prop: Vec<Vec<f64>> = Vec::new();
    let mut threshold = Vec::new();
    let mut step = Vec::new();
    let mut accu_regret = Vec::new();
    let mut thre: f64 = 1e-10;
    let mut k = 1;
    for i in tqdm(0..epoches) {
        if i % k == 0 {
            if k * 10 <= i {
                k *= 10;
            }
            let mut ugs = ugs.clone();
            let mut spc = spc.clone();
            let mut rmc = rmc.clone();
            let mut tmp_accu_regret = vec![];
            for p in 0..static_trees.num_player {
                for k in &record_accu_regret[p] {
                    tmp_accu_regret.push(max_index4vec(&rmc.get_accu_regret(p, *k)).0);
                }
            }
            accu_regret.push(tmp_accu_regret);
            spc.reset_compensate_for_test();
            rm_compensation(static_trees, &ugs, &mut rmc, &mut spc, config);
            reset_gamestate_root(static_trees, &mut ugs, vec![0], config);
            step.push(vec![i as f64]);
            let expected = compute_expect_return(&static_trees, &mut ugs, &rmc, &spc, config);
            let bestres = compute_br_return(&static_trees, &mut ugs, &rmc, config);
            exploitablity.push(vec![
                (bestres.iter().sum::<f64>() - expected.iter().sum::<f64>())
                    / static_trees.num_player as f64,
            ]);
            threshold.push(vec![thre]);
            let prop = spc.get_total_pruned_nodes();
            prune_prop.push(vec![
                prop.0 .0 as f64
                    / (static_trees.trees.0.len() - {
                        if config.count_leaves {
                            0
                        } else {
                            static_trees.num_publeaves()
                        }
                    }) as f64
                    / config.get_current_iteration() as f64,
                prop.1 .0 as f64
                    / (static_trees.trees.0.len() - {
                        if config.count_leaves {
                            0
                        } else {
                            static_trees.num_publeaves()
                        }
                    }) as f64
                    / config.get_current_iteration() as f64,
                prop.0 .1 as f64
                    / (static_trees.trees.2.len() - {
                        if config.count_leaves {
                            0
                        } else {
                            static_trees.num_gamestate_leaves()
                        }
                    }) as f64
                    / config.get_current_iteration() as f64,
                prop.1 .1 as f64
                    / (static_trees.trees.2.len() - {
                        if config.count_leaves {
                            0
                        } else {
                            static_trees.num_gamestate_leaves()
                        }
                    }) as f64
                    / config.get_current_iteration() as f64,
            ]);
            expected_returns.push(expected.clone());
            bestres_returns.push(bestres.clone());
        }
        cfr_iteration_stage1(&static_trees, ugs, &rmc, &spc, false, config);
        if config.subgame_prune.0 {
            spc.update_spc_cells(static_trees, ugs, config); // update history_prob_r (realization_plan)
            if config.subgame_prune.1 {
                check_free_subgame_checking(static_trees, ugs, rmc, &mut spc, config);
            }
            let (first_pruned_roots, compensate_roots) =
                prune_constraint_checking(static_trees, ugs, rmc, &mut spc, config);
            rm_compensation(static_trees, ugs, rmc, &mut spc, config);
            reset_gamestate_root(static_trees, ugs, compensate_roots, config);
            set_probr2h_and_root(static_trees, ugs, rmc, &spc, first_pruned_roots, config);
        }
        cfr_iteration_stage2(&static_trees, &ugs, rmc, &spc, config);
        cfr_iteration_stage3(&static_trees, &ugs, rmc, &spc, config);
        thre = config.step();
    }
    /*
    // plot
    let mut data = HashMap::new();
    data.insert(
        "iteration".to_string(),
        step.clone().into_iter().flatten().collect::<Vec<f64>>(),
    );
    data.insert(
        "exploitablity".to_string(),
        exploitablity
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>(),
    );
    data.insert(
        "threshold".to_string(),
        threshold
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>(),
    );
    for p in 0..static_trees.num_player {
        let mut k = 0;
        for infoset_index in record_accu_regret[p].iter() {
            data.insert(
                format!("p{}-index{}", p, infoset_index),
                accu_regret.iter().map(|x| x[k]).collect(),
            );
            k += 1;
        }
    }
    let _ = plot_loss_curves(
        data,
        &(format!("{}/plot/", config.dir)
            + &config.game_name
            + "/"
            + &config.algo_name
            + "/"
            + &config.to_string()
            + "accumulated_regret.png"),
        false,
        "Accumulated Regret Curves",
        "accumulated regret",
    );

    let mut data = HashMap::new();
    data.insert(
        "iteration".to_string(),
        step.clone().into_iter().flatten().collect::<Vec<f64>>(),
    );
    data.insert(
        "exploitablity".to_string(),
        exploitablity
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>(),
    );
    data.insert(
        "threshold".to_string(),
        threshold
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>(),
    );
    let _ = plot_loss_curves(
        data,
        &(format!("{}/plot/", config.dir)
            + &config.game_name
            + "/"
            + &config.algo_name
            + "/"
            + &config.to_string()
            + "exploitablity_threshold.png"),
        false,
        "Exploitablity & Dynamic Threshold",
        "exploitability/dynamic threshold",
    ); */

    // write into CSV
    let headers = vec![
        "Itera",
        "Exploitablity",
        "Expected_return-0",
        "Expected_return-1",
        "BR-0",
        "BR-1",
        "PrunedProp-pubnode",
        "CheckFreePRrop-pubnode",
        "PrunedProp-gamestate",
        "CheckFreePRrop-gamestate",
    ];
    let data = vec![
        step,
        exploitablity,
        expected_returns,
        bestres_returns,
        prune_prop,
    ];
    let data = transpose(&flatten_outer_vec(data));
    let _ = write_csv(
        &(format!("{}/csv/", config.dir)
            + &config.game_name
            + "/"
            + &config.algo_name
            + "/"
            + &config.to_string()
            + ".csv"),
        &headers,
        data,
    );
}
