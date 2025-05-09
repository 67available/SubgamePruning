use super::{
    cfr_iteration_v2::get_prob_r4pubnode,
    cfr_util::{self, check_nan},
    regret_matching,
    subgame_prune_v2::{PruneRMCell, SubgamePruneCell},
};
use crate::config;
use cfr_util::{max_index4vec, multiple, multiple_, transpose};
use config::Config;
use game_tree::tree::tree_struct::{build_ugs, UtilityGameState};
use regret_matching::{build_rmc, RMCellTrait};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

#[derive(Clone)]
pub struct SubgamePruneAgent {
    spc: HashMap<usize, SubgamePruneCell>,
    pruned_roots: HashSet<usize>,
    pruned_nodes: HashMap<usize, (usize, usize)>,
    check_free_roots: HashSet<usize>,
    check_free_nodes: HashMap<usize, (usize, usize)>,
    compensate_roots: HashSet<usize>,
    compensate_nodes: HashMap<usize, (usize, usize)>,
    total_prune_nodes: (usize, usize),
    total_check_free_nodes: (usize, usize),
}

impl SubgamePruneAgent {
    pub fn new() -> Self {
        SubgamePruneAgent {
            spc: HashMap::new(),
            pruned_roots: HashSet::new(),
            pruned_nodes: HashMap::new(),
            check_free_roots: HashSet::new(),
            check_free_nodes: HashMap::new(),
            compensate_roots: HashSet::new(),
            compensate_nodes: HashMap::new(),
            total_prune_nodes: (0, 0),
            total_check_free_nodes: (0, 0),
        }
    }
    pub fn get_times(&self, pubnode_index: usize) -> usize {
        let (a, b) = self.check_under_pruned(pubnode_index);
        assert!(a);
        self.spc.get(&b).unwrap().prune_times
    }
    pub fn get_history_prob_r4pubnode(
        &self,
        pubnode_index: usize,
        avr: bool,
    ) -> HashMap<usize, Vec<f64>> {
        if avr {
            self.spc
                .get(&pubnode_index)
                .unwrap()
                .prob_r_history4avr_update
                .clone()
        } else {
            self.spc.get(&pubnode_index).unwrap().prob_r_history.clone()
        }
    }
    pub fn update_spc_cells(
        &mut self,
        static_trees: &StaticTrees,
        ugs: &UgsAgent,
        config: &Config,
    ) {
        let cfrplus = config.algo_name == "CFR+".to_string();
        for pubnode_index in self.pruned_roots.iter() {
            let prob_r = get_prob_r4pubnode(*pubnode_index, static_trees, ugs, None);
            self.spc
                .get_mut(pubnode_index)
                .unwrap()
                .update_prob_r_history(prob_r, cfrplus);
        }
    }
    pub fn pruned_roots_contain(&self, pubnode_index: usize) -> bool {
        self.pruned_roots.contains(&pubnode_index)
    }

    pub fn whether_check_free(&self, pubnode_index: usize) -> (bool, usize) {
        let mut root = 0;
        let mut res = false;
        for (root_index, x) in self.check_free_nodes.iter() {
            if x.0 <= pubnode_index && pubnode_index <= x.1 {
                res = true;
                root = *root_index;
                break;
            }
        }
        (res, root)
    }
    pub fn check_check_free_node(
        &self,
        pubnode_index: usize,
        prob_r: &HashMap<usize, Vec<f64>>,
        config: &Config,
    ) -> bool {
        self.spc
            .get(&pubnode_index)
            .unwrap()
            .check_exempt_condition(prob_r, config)
    }
    pub fn get_pruned_subgame(&self) -> HashSet<usize> {
        self.pruned_roots.clone()
    }
    pub fn check_under_pruned(&self, pubnode_index: usize) -> (bool, usize) {
        let mut root = 0;
        let mut res = false;
        for (root_index, x) in self.pruned_nodes.iter() {
            if x.0 <= pubnode_index && pubnode_index <= x.1 {
                res = true;
                root = *root_index;
                break;
            }
        }
        (res, root)
    }
    pub fn check_under_compensated(&self, pubnode_index: usize) -> (bool, usize) {
        let mut root = 0;
        let mut res = false;
        for (root_index, x) in self.compensate_nodes.iter() {
            if x.0 <= pubnode_index && pubnode_index <= x.1 {
                res = true;
                root = *root_index;
                break;
            }
        }
        (res, root)
    }
    pub fn get_compensation_roots(&self) -> HashSet<usize> {
        self.compensate_roots.clone()
    }
    pub fn update_check_free_roots(
        &mut self,
        check_free_roots: HashSet<usize>,
        static_trees: &StaticTrees,
        config: &Config,
    ) {
        self.check_free_roots = check_free_roots;
        let mut move_out_keys = vec![];
        self.check_free_nodes.keys().for_each(|pubnode_index| {
            if !self.check_free_roots.contains(pubnode_index) {
                move_out_keys.push(*pubnode_index);
            }
        });
        move_out_keys.iter().for_each(|k| {
            self.check_free_nodes.remove(k);
        });
        self.check_free_roots.iter().for_each(|pubnode_index| {
            if !self.check_free_nodes.contains_key(pubnode_index) {
                self.check_free_nodes.insert(
                    *pubnode_index,
                    self.pruned_nodes.get(pubnode_index).unwrap().clone(),
                );
            }
        });
        self.check_free_nodes.iter().for_each(|(_, (a, b))| {
            self.total_check_free_nodes.0 += b - a + 1;
            for pubnode_index in *a..(*b + 1) {
                self.total_check_free_nodes.1 +=
                    static_trees.trees.0[pubnode_index].gamestates.len();
            }
        });
        if !config.count_leaves {
            let depth = static_trees.get_depth();
            for i in 0..depth {
                static_trees.get_leaves_at_level(i).iter().for_each(|&x| {
                    if self.whether_check_free(x).0 {
                        self.total_check_free_nodes.0 -= 1;
                        self.total_check_free_nodes.1 -= static_trees.trees.0[x].gamestates.len();
                    }
                });
            }
        }
    }

    pub fn update_pruned_roots<T: RMCellTrait + Sized + Sync + Send + PruneRMCell>(
        &mut self,
        pruned_roots: Vec<usize>,
        static_trees: &StaticTrees,
        ugs: &mut UgsAgent,
        rmc: &RMAgent<T>,
        config: &Config,
    ) -> (Vec<usize>, Vec<usize>) {
        /* Identify newly pruned root nodes and root nodes that require compensation */
        self.compensate_roots.clear();
        self.compensate_nodes.clear();
        self.pruned_roots.iter().for_each(|x| {
            if !pruned_roots.contains(x) {
                self.compensate_roots.insert(*x);
                self.compensate_nodes
                    .insert(*x, (*x, static_trees.trees.0[*x].end_of_subtree));
            }
        });
        let mut first_pruned_roots = vec![];
        for &pubnode_index in pruned_roots.iter() {
            if !self.spc.contains_key(&pubnode_index) {
                first_pruned_roots.push(pubnode_index);
                let mut max_regret_in_subgame = 0.0;
                for p in 0..static_trees.num_player {
                    static_trees.trees.0[pubnode_index]
                        .infosets
                        .get(&p)
                        .unwrap()
                        .iter()
                        .for_each(|&infoset_index| {
                            if rmc.get_max_regret_in_subgame(p, infoset_index)
                                > max_regret_in_subgame
                            {
                                max_regret_in_subgame =
                                    rmc.get_max_regret_in_subgame(p, infoset_index)
                            }
                        });
                }
                let prob_r_before_prune =
                    get_prob_r4pubnode(pubnode_index, static_trees, ugs, None);
                let new_spc = SubgamePruneCell::new(
                    &prob_r_before_prune,
                    max_regret_in_subgame,
                    config.get_current_iteration(),
                );
                self.spc.insert(pubnode_index, new_spc);
                self.pruned_nodes.insert(
                    pubnode_index,
                    (
                        pubnode_index,
                        static_trees.trees.0[pubnode_index].end_of_subtree,
                    ),
                );
            }
        }
        self.pruned_roots = HashSet::from_iter(pruned_roots.into_iter());
        self.pruned_nodes.iter().for_each(|(_, (a, b))| {
            self.total_prune_nodes.0 += b - a + 1;
            for pubnode_index in *a..(*b + 1) {
                self.total_prune_nodes.1 += static_trees.trees.0[pubnode_index].gamestates.len();
            }
        });
        if !config.count_leaves {
            assert!(false);
            let depth = static_trees.get_depth();
            for i in 0..depth {
                static_trees.get_leaves_at_level(i).iter().for_each(|&x| {
                    if self.check_under_pruned(x).0 {
                        self.total_prune_nodes.0 -= 1;
                        self.total_prune_nodes.1 -= static_trees.trees.0[x].gamestates.len();
                    }
                });
            }
        }
        (
            first_pruned_roots,
            self.compensate_roots.iter().map(|&x| x).collect(),
        )
    }
    pub fn remove(&mut self, pubnode_index: usize) {
        if self.check_free_roots.contains(&pubnode_index) {
            self.check_free_roots.remove(&pubnode_index);
        }
        self.check_free_nodes.remove(&pubnode_index);
        self.pruned_nodes.remove(&pubnode_index);
        self.pruned_roots.remove(&pubnode_index);
        self.spc.remove(&pubnode_index);
    }
    pub fn get_total_pruned_nodes(&self) -> ((usize, usize), (usize, usize)) {
        (self.total_prune_nodes, self.total_check_free_nodes)
    }
    pub fn reset_compensate_for_test(&mut self) {
        self.compensate_roots = self.pruned_roots.clone();
    }
}

#[derive(Clone)]
pub struct UgsAgent {
    pub ugs: Vec<UtilityGameState>,
    game_state_children_indices: Vec<Vec<usize>>,
    current_p: Vec<usize>,
}

impl UgsAgent {
    pub fn new(
        trees: &(
            Vec<game_tree::tree::tree_struct::PubNode>,
            HashMap<usize, Vec<game_tree::tree::tree_struct::InfoSet>>,
            Vec<game_tree::tree::tree_struct::GameState>,
        ),
    ) -> Self {
        UgsAgent {
            ugs: build_ugs(&trees.2),
            game_state_children_indices: trees
                .2
                .iter()
                .map(|x| x.children.clone())
                .collect::<Vec<Vec<usize>>>(),
            current_p: trees
                .2
                .iter()
                .map(|x| x.player.unwrap_or(2))
                .collect::<Vec<usize>>(),
        }
    }
    pub fn update_children_rp(&mut self, index: usize, strategy: &Vec<f64>) {
        /*
        // Discuss four cases:
        // 1. Neither the parent nor the child node is pruned.
        // 2. The child node is pruned, but the parent is not.
        // 3. Both are pruned, and the parent is the root of a pruned subgame.
        // 4. Both are pruned, but neither is the root of a pruned subgame.
        // These cases can be handled separately and then merged accordingly.
         */
        assert_eq!(
            self.game_state_children_indices[index].len(),
            strategy.len()
        );
        for (k, child_index) in self.game_state_children_indices[index].iter().enumerate() {
            self.ugs[*child_index].prob_r = self.ugs[index].prob_r.clone();
            self.ugs[*child_index].prob_r2h = self.ugs[index].prob_r2h.clone();
            if self.ugs[*child_index].root == *child_index && self.ugs[index].root == index {
                self.ugs[*child_index].prob_r[self.current_p[index]] *= strategy[k];
            } else {
                self.ugs[*child_index].prob_r2h[self.current_p[index]] *= strategy[k];
            }
        }
    }

    pub fn update_self_utitlity(&mut self, index: usize, strategy: &Vec<f64>) {
        let u = self.game_state_children_indices[index]
            .iter()
            .enumerate()
            .map(|(_action_index, &x)| self.ugs[x].utitlity.clone())
            .collect::<Vec<Vec<f64>>>();
        self.ugs[index].utitlity = multiple_(&transpose(&u), strategy);
    }
    pub fn get_realization_plan4p(
        &self,
        gamestate_index: usize,
        player: usize,
        subgame_prune: bool,
        history_prob_r: Option<&HashMap<usize, Vec<f64>>>,
    ) -> f64 {
        if subgame_prune {
            self.ugs[gamestate_index].prob_r2h[player]
                * history_prob_r
                    .unwrap()
                    .get(&self.ugs[gamestate_index].root)
                    .unwrap()[player]
        } else {
            self.ugs[gamestate_index].prob_r[player] * self.ugs[gamestate_index].prob_r2h[player]
        }
    }
    pub fn get_realization_plan_exclude_p(
        &self,
        gamestate_index: usize,
        player2exclude: usize,
        subgame_prune: bool,
        history_prob_r: Option<&HashMap<usize, Vec<f64>>>,
    ) -> f64 {
        let mut res = 1.0;
        for p in 0..self.ugs[gamestate_index].prob_r.len() {
            if p != player2exclude {
                if subgame_prune && p != self.ugs[gamestate_index].prob_r.len() - 1 {
                    res *= history_prob_r
                        .unwrap()
                        .get(&self.ugs[gamestate_index].root)
                        .unwrap()[p]
                        * self.ugs[gamestate_index].prob_r2h[p];
                    assert!(!res.is_nan());
                } else {
                    res *=
                        self.ugs[gamestate_index].prob_r[p] * self.ugs[gamestate_index].prob_r2h[p];
                    assert!(!res.is_nan());
                }
            }
        }
        res
    }
    pub fn get_utitliy_vec(&self, index: usize) -> Vec<f64> {
        self.ugs[index].utitlity.clone()
    }
    pub fn update_gamestate_from_root(&mut self, root_gamestate_index: usize) {
        let mut quque = VecDeque::new();
        quque.push_back(root_gamestate_index);
        while quque.len() > 0 {
            let level_size = quque.len();
            for _ in 0..level_size {
                let gamestate_index = quque.pop_front().unwrap();
                self.game_state_children_indices[gamestate_index]
                    .iter()
                    .for_each(|x| quque.push_back(*x));
                self.ugs[gamestate_index].root = root_gamestate_index;
            }
        }
    }
    pub fn reset_gamestate_from_root(&mut self, root_gamestate_index: usize) {
        let mut quque = VecDeque::new();
        quque.push_back(root_gamestate_index);
        while quque.len() > 0 {
            let level_size = quque.len();
            for _ in 0..level_size {
                let gamestate_index = quque.pop_front().unwrap();
                self.game_state_children_indices[gamestate_index]
                    .iter()
                    .for_each(|x| quque.push_back(*x));
                self.ugs[gamestate_index].root = gamestate_index;
            }
        }
    }
}

impl fmt::Display for UgsAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (gamestate_index, node) in self.ugs.iter().enumerate() {
            write!(f, "index {} {}", gamestate_index, node)?;
            writeln!(
                f,
                "children {:?}",
                self.game_state_children_indices[gamestate_index]
            )?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct RMAgent<T: RMCellTrait + Sized + PruneRMCell> {
    pub rmc: HashMap<usize, Vec<T>>,
    children: HashMap<usize, HashMap<usize, Vec<usize>>>,
    pub algo_name: String,
}

fn find_narrow_children_infoset(
    trees: &(
        Vec<game_tree::tree::tree_struct::PubNode>,
        HashMap<usize, Vec<game_tree::tree::tree_struct::InfoSet>>,
        Vec<game_tree::tree::tree_struct::GameState>,
    ),
    p: usize,
    infoset_index: usize,
) -> Vec<usize> {
    let mut queue = VecDeque::new();
    trees.1.get(&p).unwrap()[infoset_index]
        .children
        .iter()
        .for_each(|&x| queue.push_back(x));
    while queue.len() != 0 {
        let level_size = queue.len();
        let mut level_nodes = vec![];
        for _ in 0..level_size {
            level_nodes.push(queue.pop_front().unwrap());
        }
        if trees.1.get(&p).unwrap()[level_nodes[0]].player == p as i32 {
            return level_nodes;
        }
        level_nodes.into_iter().for_each(|x| {
            trees.1.get(&p).unwrap()[x]
                .children
                .iter()
                .for_each(|&y| queue.push_back(y))
        });
    }
    return vec![];
}

impl<T: RMCellTrait + Sized + PruneRMCell> RMAgent<T> {
    pub fn new(
        trees: &(
            Vec<game_tree::tree::tree_struct::PubNode>,
            HashMap<usize, Vec<game_tree::tree::tree_struct::InfoSet>>,
            Vec<game_tree::tree::tree_struct::GameState>,
        ),
    ) -> Self {
        let rmc = trees
            .1
            .iter()
            .map(|(x, y)| return (*x, build_rmc(y)))
            .collect::<HashMap<usize, Vec<T>>>();
        let mut children = HashMap::new();
        for p in 0..trees.1.len() {
            children.insert(p, HashMap::new());
            trees
                .1
                .get(&p)
                .unwrap()
                .iter()
                .enumerate()
                .for_each(|(index, &ref infoset)| {
                    children
                        .get_mut(&p)
                        .unwrap()
                        .insert(index, find_narrow_children_infoset(trees, p, index));
                });
        }
        RMAgent {
            rmc,
            children,
            algo_name: T::ALGO_NAME.to_string(),
        }
    }
    pub fn update_avr_strategy(
        &mut self,
        p: usize,
        infoset_index: usize,
        weight: f64,
        times: usize,
    ) {
        self.rmc.get_mut(&p).unwrap()[infoset_index].avr_update(weight, times);
    }
    pub fn update_accu_regret(&mut self, p: usize, infoset_index: usize, regret: &Vec<f64>) {
        self.rmc.get_mut(&p).unwrap()[infoset_index].rm(regret);
    }
    pub fn get_accu_regret(&self, p: usize, infoset_index: usize) -> Vec<f64> {
        self.rmc.get(&p).unwrap()[infoset_index].get_accu_regret()
    }
    pub fn set_max_regret4subgame(&mut self, p: usize, infoset_index: usize, max_regret: f64) {
        let mut children_max_regret = self
            .children
            .get(&p)
            .unwrap()
            .get(&infoset_index)
            .unwrap()
            .iter()
            .map(|child_index| self.rmc.get(&p).unwrap()[*child_index].get_max_regret())
            .collect::<Vec<f64>>();
        children_max_regret.push(max_regret);
        self.rmc.get_mut(&p).unwrap()[infoset_index]
            .set_max_regret(max_index4vec(&children_max_regret).0);
    }
    pub fn prune_constraint_check(
        &self,
        p: usize,
        infoset_index: usize,
        regret: &Vec<f64>,
        config: &Config,
    ) -> bool {
        self.rmc.get(&p).unwrap()[infoset_index].prune_constraint_check(regret, config)
    }
    pub fn get_max_regret_in_subgame(&self, p: usize, infoset_index: usize) -> f64 {
        if self.rmc.get(&p).unwrap()[infoset_index].get_max_regret() == f64::MAX {
            let children_max_regret = self
                .children
                .get(&p)
                .unwrap()
                .get(&infoset_index)
                .unwrap()
                .iter()
                .map(|child_index| self.rmc.get(&p).unwrap()[*child_index].get_max_regret())
                .collect::<Vec<f64>>();
            if children_max_regret.len() == 0 {
                return 0.0;
            }
            max_index4vec(&children_max_regret).0
        } else {
            self.rmc.get(&p).unwrap()[infoset_index].get_max_regret()
        }
    }
}

pub struct StaticTrees {
    pub trees: (
        Vec<game_tree::tree::tree_struct::PubNode>,
        HashMap<usize, Vec<game_tree::tree::tree_struct::InfoSet>>,
        Vec<game_tree::tree::tree_struct::GameState>,
    ),
    pub num_player: usize,
    leaves_by_level: Vec<Vec<usize>>,
}

impl StaticTrees {
    pub fn new(
        trees: (
            Vec<game_tree::tree::tree_struct::PubNode>,
            HashMap<usize, Vec<game_tree::tree::tree_struct::InfoSet>>,
            Vec<game_tree::tree::tree_struct::GameState>,
        ),
    ) -> Self {
        let num_player = trees.1.len();
        let mut leaves_by_level = vec![];
        let mut queue = VecDeque::new();
        queue.push_back(0);
        while !queue.is_empty() {
            let mut leaves = vec![];
            let level_size = queue.len();
            let mut level_nodes = Vec::with_capacity(level_size);
            for _ in 0..level_size {
                let index = queue.pop_front().unwrap();
                let node = &trees.0[index];
                level_nodes.push(index);
                for &child_index in &node.children {
                    queue.push_back(child_index);
                }
                if node.children.len() == 0 {
                    leaves.push(index);
                }
            }
            leaves_by_level.push(leaves);
        }
        StaticTrees {
            trees,
            num_player,
            leaves_by_level,
        }
    }

    pub fn get_pubnode_type(&self, node_index: usize) -> String {
        self.trees.2[self.trees.0[node_index].gamestates[0]]
            .node_type
            .clone()
    }
    pub fn get_pubnode_player(&self, node_index: usize) -> usize {
        self.trees.2[self.trees.0[node_index].gamestates[0]]
            .player
            .clone()
            .unwrap_or(self.num_player + 1)
    }
    pub fn get_depth(&self) -> usize {
        self.leaves_by_level.len()
    }
    pub fn get_leaves_at_level(&self, level: usize) -> Vec<usize> {
        return self.leaves_by_level[level].clone();
    }
    pub fn get_all_nodes_rooted_at_p(&self, pubnode_index: usize) -> Vec<usize> {
        Vec::from_iter(pubnode_index..self.trees.0[pubnode_index].end_of_subtree)
    }
    pub fn num_publeaves(&self) -> usize {
        self.leaves_by_level.iter().map(|x| x.len()).sum()
    }
    pub fn num_gamestate_leaves(&self) -> usize {
        self.leaves_by_level
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| self.trees.0[*y].gamestates.len())
                    .sum::<usize>()
            })
            .sum()
    }
}

pub fn get_cv4infoset(
    static_trees: &StaticTrees,
    ugs: &UgsAgent,
    p: usize,
    rmc_index: usize,
    history_prob_r: Option<&HashMap<usize, Vec<f64>>>,
) -> f64 {
    /*
    calculate counterfactual value for info-set
     */
    let subgame_prune = history_prob_r.is_some();
    if subgame_prune {
        history_prob_r.unwrap();
    }
    let outsize_realization_plan = static_trees.trees.1[&p][rmc_index]
        .i2state
        .iter()
        .map(|&x| ugs.get_realization_plan_exclude_p(x, p, subgame_prune, history_prob_r))
        .collect::<Vec<f64>>();
    let u = static_trees.trees.1[&p][rmc_index]
        .i2state
        .iter()
        .map(|&x| ugs.get_utitliy_vec(x)[p])
        .collect::<Vec<f64>>();
    check_nan(&outsize_realization_plan);
    check_nan(&u);
    let cv_i = multiple(&u, &outsize_realization_plan);
    return cv_i;
}

pub fn get_br_strategy4pubnode(
    static_trees: &StaticTrees,
    ugs: &UgsAgent,
    node_index: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let node = &static_trees.trees.0[node_index];
    let node_type = static_trees.get_pubnode_type(node_index);
    assert_eq!(node_type, "P".to_string());
    let actor = static_trees.get_pubnode_player(node_index);
    let mut strategy_matrix: Vec<Vec<f64>> = vec![];
    let mut game_state_cluster = vec![];
    for rmc_index in static_trees.trees.0[node_index]
        .infosets
        .get(&actor)
        .unwrap()
        .iter()
    {
        let cv_ia = static_trees.trees.1[&actor][*rmc_index]
            .children
            .iter()
            .map(|&x| get_cv4infoset(static_trees, ugs, actor, x, None))
            .collect::<Vec<f64>>();
        let (_, best_action_index) = max_index4vec(&cv_ia);
        let br_strategy = (0..static_trees.trees.1.get(&actor).unwrap()[*rmc_index]
            .children
            .len())
            .map(|action_index| {
                if action_index == best_action_index {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        strategy_matrix.push(br_strategy);
        game_state_cluster.push(
            static_trees.trees.1.get(&actor).unwrap()[*rmc_index]
                .i2state
                .clone(),
        );
    }
    return (strategy_matrix, game_state_cluster);
}

pub fn get_strategy4pubnode<T: RMCellTrait + Sized + PruneRMCell>(
    static_trees: &StaticTrees,
    rmc: &RMAgent<T>,
    node_index: usize,
    avr: bool, // true: return the average strategy; false: return the current strategy
) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    /* get strategy list in the public node and corresponding game states */
    let node = &static_trees.trees.0[node_index];
    let node_type = static_trees.get_pubnode_type(node_index);
    let actor = static_trees.get_pubnode_player(node_index);
    let mut strategy_matrix: Vec<Vec<f64>> = vec![];
    let mut game_state_cluster = vec![];
    if node_type == 'C'.to_string() {
        strategy_matrix = node
            .gamestates
            .iter()
            .map(|&x| static_trees.trees.2[x].chance.clone().unwrap())
            .collect();
        node.gamestates
            .iter()
            .for_each(|x| game_state_cluster.push(vec![*x]));
    } else if node_type == 'T'.to_string() {
    } else {
        for &i in node.infosets[&actor].iter() {
            if avr {
                strategy_matrix.push(rmc.rmc[&actor][i].get_avr());
            } else {
                strategy_matrix.push(rmc.rmc[&actor][i].get_cur());
            }
            game_state_cluster.push(static_trees.trees.1[&actor][i].i2state.clone());
        }
    }
    assert_eq!(strategy_matrix.len(), game_state_cluster.len());
    return (strategy_matrix, game_state_cluster);
}
