use crate::{
    br,
    dfs::{PruneVisitor, dfs_with_pruning_backprop},
    load_tree::{History, InfoSet, PubNode, from_json},
    logger::Logger,
    timer::CfrTimerWithSamples,
};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Instant,
};

fn has_nan(values: &Vec<f64>) -> bool {
    values.iter().any(|v| v.is_nan())
}

struct RegretMatchingCell {
    strategy: Vec<f64>,
    avg_strategy: Vec<f64>,
    accu_regret: Vec<f64>,
    avg_weight: f64,      // 平均策略权重
    ins_regret: Vec<f64>, // 瞬时遗憾
}

impl RegretMatchingCell {
    fn new(num_action: usize) -> Self {
        RegretMatchingCell {
            strategy: vec![1.0 / num_action as f64; num_action],
            avg_strategy: vec![1.0 / num_action as f64; num_action],
            accu_regret: vec![0.0; num_action],
            avg_weight: 0.0,
            ins_regret: vec![0.0; num_action],
        }
    }
}

pub struct Policy {
    inner: HashMap<Arc<String>, RegretMatchingCell>,
}

impl Policy {
    pub fn get_strategy(&self, infoset_str: &String) -> Vec<f64> {
        self.inner[infoset_str].strategy.clone()
    }
    pub fn get_avg_strategy(&self, infoset_str: &String) -> Vec<f64> {
        self.inner[infoset_str].avg_strategy.clone()
    }
    pub fn update_regret(&mut self, infoset_str: &String, regret: &Vec<f64>, cfr_plus: bool) {
        self.inner
            .get_mut(infoset_str)
            .unwrap()
            .accu_regret
            .iter_mut()
            .zip(regret.iter())
            .for_each(|(x, y)| {
                *x += y;
                if cfr_plus && *x < 0.0 {
                    *x = 0.0
                }
            });
    }

    pub fn update_avg_strategy(&mut self, infoset_str: &String, weight: f64) {
        let rmcell = self.inner.get_mut(infoset_str).unwrap();
        rmcell
            .avg_strategy
            .iter_mut()
            .zip(rmcell.strategy.iter())
            .for_each(|(x, y)| {
                *x = (*x * rmcell.avg_weight + y * weight) / (rmcell.avg_weight + weight)
            });
        rmcell.avg_weight += weight;
        // dbg!(&rmcell.avg_weight);
    }
    pub fn update_current_strategy(&mut self, infoset_str: &String) {
        let rmcell = self.inner.get_mut(infoset_str).unwrap();
        let m: f64 = rmcell
            .accu_regret
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .sum();
        if m > 0.0 {
            rmcell.strategy = rmcell
                .accu_regret
                .iter()
                .map(|&x| if x > 0.0 { x / m } else { 0.0 })
                .collect();
        } else {
            rmcell.strategy = vec![1.0 / rmcell.strategy.len() as f64; rmcell.strategy.len()];
        }
    }

    // subgame_prune
    pub fn check_regret(&self, infoset_str: &String, regret: &Vec<f64>) -> bool {
        self.inner
            .get(infoset_str)
            .unwrap()
            .accu_regret
            .iter()
            .zip(regret.iter())
            .map(|(&x, &y)| if x >= 0.0 { y.abs() == 0.0 } else { y < -x })
            .all(|x| x)
    }
}

pub fn build_policy(
    static_trees: &(
        Vec<PubNode>,
        Vec<Vec<InfoSet>>,
        Vec<HashMap<String, InfoSet>>,
        Vec<History>,
    ),
) -> Vec<Policy> {
    let num_player = static_trees.1.len();
    let mut policy_vec = vec![];
    for p in 0..num_player {
        // 并行生成 Vec<(Arc<_>, RegretMatchingCell)>
        let map: HashMap<Arc<String>, RegretMatchingCell> = static_trees.1[p]
            .par_iter()
            .filter_map(|x| {
                (static_trees.3[x.i2history[0]].current_player == p as i32).then(|| {
                    (
                        Arc::new(x.key.clone()),
                        RegretMatchingCell::new(x.children.len()),
                    )
                })
            })
            .collect();
        dbg!("num decision_node {}", map.len());
        policy_vec.push(Policy { inner: map });
    }
    policy_vec
}

/// 与history一一对应
struct UtilityCell {
    alpha_history_idx: Option<usize>, // 被裁减子博弈的根节点中为当前历史的祖先节点的历史
    rp_root2alpha_on_prune: Option<Vec<f64>>, // 记录被裁剪时，从根节点到子博弈根节点的rp，对于subgame内的节点通过rp/rp_root2alpha_on_prune即可获得rp_alpha2h_on_prune，这么做是为了避免为了更新rp_alpha2h_on_prune特地又遍历一次子树。
    rp_alpha2h_on_prune: Option<Vec<f64>>,    // 这两个属性一但算好后就不再变了
    accum_rp: Option<Vec<f64>>,               // 被裁剪的子博弈的根节点的累计到达概率
    last_rp: Option<Vec<f64>>,                //上一次做rp更新时的rp
    rp: Vec<f64>,                             // 被裁减的节点就无需更新rp了
    utility: Vec<f64>,                        // 各个玩家的收益
}

impl UtilityCell {
    fn new(num_player: usize) -> Self {
        UtilityCell {
            alpha_history_idx: None,
            rp_root2alpha_on_prune: None,
            rp_alpha2h_on_prune: None,
            accum_rp: None,
            last_rp: None,
            rp: vec![1.0; num_player + 1],
            utility: vec![0.0; num_player],
        }
    }
}

pub struct UtilityTree {
    inner: Vec<UtilityCell>, // 与game_state_tree里的history一一对应
}

impl UtilityTree {
    // -------------------------------------- vanilla cfr -----------------------------------//
    /// 子节点idx与strategy必须是一一对应的
    pub fn update_rp(
        &mut self,
        father: usize,
        children: &[usize],
        strategy: &[f64],
        current_player: usize,
    ) {
        for (&child, p) in children.iter().zip(strategy.iter()) {
            // 1. 先从父节点复制rp，赋值到子节点的rp上
            self.inner[child].rp = self.inner[father].rp.clone();
            // 2. 再使用strategy更新rp
            self.inner[child].rp[current_player] *= p;
        }
    }

    pub fn update_utility(&mut self, father: usize, children: &[usize], strategy: &[f64]) {
        // 1. 对子节点的utility做线性加权后赋值到当前节点
        self.inner[father].utility = {
            let n_players = self.inner[father].utility.len();
            let init = vec![0.0f64; n_players];

            children
                .iter()
                .zip(strategy.iter())
                .fold(init, |mut acc, (&child, &prob)| {
                    let u = &self.inner[child].utility; // &Vec<f64>
                    for i in 0..acc.len() {
                        acc[i] += prob * u[i];
                    }
                    acc
                })
        };
        if has_nan(&self.inner[father].utility) {
            dbg!(&self.inner[father].utility);
        }
    }

    pub fn update_leaf_utility(&mut self, idx: usize, utiltity: &[f64]) {
        self.inner[idx].utility = utiltity.to_vec();
    }

    pub fn get_rp_outside(&self, current_player: usize, idx: usize) -> f64 {
        self.inner[idx]
            .rp
            .iter()
            .enumerate()
            .fold(1.0, |product, (player, rp)| {
                if player != current_player {
                    product * rp
                } else {
                    product
                }
            })
    }

    pub fn get_self_rp(&self, current_player: usize, idx: usize) -> f64 {
        self.inner[idx].rp[current_player]
    }

    pub fn get_utility_p(&self, current_player: usize, idx: usize) -> f64 {
        self.inner[idx].utility[current_player]
    }

    // ------------------------------------ subgame prune ------------------------------ //
    pub fn get_self_rp_diffence(&self, current_player: usize, idx: usize) -> f64 {
        let history = &self.inner[idx];
        (history.rp[current_player] - history.last_rp.as_ref().unwrap()[current_player]).abs()
    }

    pub fn get_self_rp_compensate(&self, current_player: usize, idx: usize) -> f64 {
        let root_of_pruned_subgame = self.inner[idx].alpha_history_idx.unwrap();
        let rp_2alpha = self.inner[root_of_pruned_subgame]
            .accum_rp
            .as_ref()
            .unwrap()[current_player];
        // self.inner[idx].rp[current_player]
        //         / self.inner[root_of_pruned_subgame].rp[current_player] // 此处不能直接除，因为分母可能为0，这样会丢失alpha2h_rp的信息
        let rp_alpha2 = self.inner[idx].rp_alpha2h_on_prune.as_ref().unwrap()[current_player];
        if rp_alpha2.is_nan() {
            panic!("解决除零问题后不应到达此处。")
        }
        rp_2alpha * rp_alpha2
    }

    pub fn get_rp_outside_compensate(&self, current_player: usize, idx: usize) -> f64 {
        let root_of_pruned_subgame = self.inner[idx].alpha_history_idx.unwrap();
        let rp_2alpha = self.inner[root_of_pruned_subgame]
            .accum_rp
            .as_ref()
            .unwrap();
        let rp_alpha2 = self.inner[idx].rp_alpha2h_on_prune.as_ref().unwrap();
        // dbg!(rp_2alpha, rp_alpha2);
        rp_alpha2
            .iter()
            .zip(rp_2alpha.iter())
            .enumerate()
            .fold(1.0, |product, (player, rp)| {
                if player != current_player {
                    product * rp.0 * rp.1
                } else {
                    product
                }
            })
    }

    /// 在rp已经被更新过的情况下，更新accum_rp，注意更新accum_rp时不需要累加Chance玩家的rp
    pub fn accum_rp(&mut self, idx: usize) {
        let utitliy_node = &mut self.inner[idx];
        let num_player = utitliy_node.rp.len() - 1;
        if let Some(accum_rp) = &mut utitliy_node.accum_rp {
            accum_rp
                .iter_mut()
                .zip(utitliy_node.rp.iter())
                .enumerate()
                .for_each(|(p, (x, y))| {
                    if p != num_player {
                        *x += y
                    }
                });
        } else {
            utitliy_node.accum_rp = Some(utitliy_node.rp.clone());
        }
        // let rp = std::mem::take(&mut utitliy_node.rp);
        // if let Some(accum) = utitliy_node.accum_rp.as_mut() {
        //     for (x, y) in accum.iter_mut().zip(rp) {
        //         *x += y;
        //     }
        // } else {
        //     utitliy_node.accum_rp = Some(rp);
        // }
    }

    pub fn update_alpha_history_idx(&mut self, father: usize, children: &[usize]) {
        for &child in children {
            self.inner[child].alpha_history_idx =
                Some(self.inner[father].alpha_history_idx.unwrap());
        }
    }

    pub fn update_alpha2h_rp(
        &mut self,
        father: usize,
        children: &[usize],
        strategy: &[f64],
        current_player: usize,
    ) {
        assert_eq!(strategy.len(), children.len());
        for (&child, p) in children.iter().zip(strategy.iter()) {
            // 1. 先从父节点复制rp，赋值到子节点的rp上
            self.inner[child].rp_alpha2h_on_prune = self.inner[father].rp_alpha2h_on_prune.clone();
            // 2. 再使用strategy更新rp
            self.inner[child].rp_alpha2h_on_prune.as_mut().unwrap()[current_player] *= p;
        }
    }

    /// 用于设置根节点的alpha2_rp，[1.0; num_player]
    pub fn set_alpha2h_rp(&mut self, idx: usize, alpha2h_rp: Vec<f64>) {
        self.inner[idx].rp_alpha2h_on_prune = Some(alpha2h_rp);
    }

    pub fn set_alpha_history_idx(&mut self, idx: usize, alpha_history_idx: usize) {
        self.inner[idx].alpha_history_idx = Some(alpha_history_idx)
    }

    pub fn reset(&mut self, idx: usize) {
        let history = &mut self.inner[idx];
        history.accum_rp = None;
        // history.last_rp = None; // 只需要把accum_rp清空就可以了，这里不能清空last_rp，因为现在的last_rp是每一轮更新rp之后更新的，所以on_exit里的last_rp是当前轮次的rp）
        // history.alpha_history_idx = None; // 这一项以及alpha2h不需要清零，这些项在compensate时会重新赋值，应该不会有冲突
        // history.rp_alpha2h_on_prune = None;
        // history.rp_root2alpha_on_prune = None;
    }

    #[inline(always)]
    pub fn update_last_rp(&mut self, idx: usize) {
        let history = &mut self.inner[idx];
        history.last_rp = Some(history.rp.clone());
    }

    #[inline(always)]
    pub fn update_all(
        &mut self,
        history_idx: usize,
        children: &[usize],
        strategy: &[f64],
        current_player: usize,
        num_player: usize,
        root_of_compensate: bool,
    ) {
        // 3️⃣ 获取父节点引用一次即可
        let parent = &mut self.inner[history_idx];

        // 若为根节点，则初始化 alpha 索引与 rp_alpha2h
        if root_of_compensate {
            parent.alpha_history_idx = Some(history_idx);
            // 提前分配一次，避免重复 Vec::new()
            parent.rp_alpha2h_on_prune = Some(vec![1.0; num_player + 1]);
        }

        // 从父节点拿出 alpha idx 和 rp_alpha2h 的引用
        let parent_alpha_idx = parent.alpha_history_idx.unwrap();
        // NOTE: borrow checker 限制，这里复制引用（非clone）以便后续使用
        let parent_rp = parent.rp_alpha2h_on_prune.as_ref().unwrap().clone();

        // 4️⃣ 单次遍历所有子节点，执行所有更新逻辑
        for (&child, &p) in children.iter().zip(strategy.iter()) {
            let child_node = &mut self.inner[child];

            // 4.1 alpha_history_idx 继承父节点
            child_node.alpha_history_idx = Some(parent_alpha_idx);

            // 4.2 计算 rp_alpha2h_on_prune（避免 clone 整个 Vec）
            // 我们只复制一份 parent_rp，然后原地更新对应玩家维度
            let mut rp_child = parent_rp.clone();
            rp_child[current_player] *= p;
            child_node.rp_alpha2h_on_prune = Some(rp_child);
        }
    }
}

pub fn build_utiltiy_tree(
    static_trees: &(
        Vec<PubNode>,
        Vec<Vec<InfoSet>>,
        Vec<HashMap<String, InfoSet>>,
        Vec<History>,
    ),
) -> UtilityTree {
    let num_player = static_trees.1.len();
    UtilityTree {
        inner: static_trees
            .3
            .par_iter()
            .map(|_| UtilityCell::new(num_player))
            .collect::<Vec<_>>(),
    }
}

/// DFS过程中子节点传递给父节点的返回值
#[derive(Clone)]
struct CfrVal {}

/// 记录在Partial Prune中会被裁剪的game states数量，由于要区分不同玩家为主玩家的遍历，所以是一个列表
#[derive(Clone)]
struct ParitalPruneVal {
    num_pruned_states: [usize; 2],
    total_states: usize,
}

/// 动态上下文 Policy与Utility Tree
struct DynamicCtx {
    pub cfg: crate::Config,
    pub step: usize,
    pub policy_vec: Vec<Policy>,
    pub utility_tree: UtilityTree,
}

impl DynamicCtx {
    pub fn get_root_value(&self, player: usize) -> f64 {
        self.utility_tree.get_utility_p(player, 0)
    }
}

/// 静态树
pub struct StaticTree {
    pub pubtree: Vec<PubNode>,
    pub infoset_dict: Vec<HashMap<String, InfoSet>>,
    pub game_state_tree: Vec<History>,
}

impl StaticTree {
    pub fn get_current_player(&self, pubnode_idx: usize) -> i32 {
        self.game_state_tree[self.pubtree[pubnode_idx].histories[0]].current_player
    }
}

struct Cfr;

impl PruneVisitor<DynamicCtx, StaticTree, CfrVal, CfrVal> for Cfr {
    fn on_enter(
        &mut self,
        idx: usize,
        _depth: usize,
        _signal: &Option<CfrVal>,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DfsAction, Option<CfrVal>) {
        // dbg!(format!("pubnode {} on enter", idx));
        let num_player = ctx.policy_vec.len();
        // 1. 更新子节点的rp
        let pubnode = &static_ctx.pubtree[idx];
        if pubnode.node_type == "P" {
            let current_player = static_ctx.get_current_player(idx) as usize;
            let infosets = &pubnode.infosets[current_player];
            let strategy_i = infosets
                .iter()
                .map(|x| ctx.policy_vec[current_player].get_strategy(x))
                .collect::<Vec<_>>();
            for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter()) {
                static_ctx.infoset_dict[current_player][infoset_str]
                    .i2history
                    .iter()
                    .for_each(|&history_idx| {
                        let children = &static_ctx.game_state_tree[history_idx].children;
                        ctx.utility_tree
                            .update_rp(history_idx, children, strategy, current_player)
                    });
            }
        } else if pubnode.node_type == "C" {
            let histories = &pubnode.histories;
            for &history_idx in histories {
                let strategy = static_ctx.game_state_tree[history_idx]
                    .chance
                    .as_ref()
                    .unwrap();
                let children = &static_ctx.game_state_tree[history_idx].children;
                ctx.utility_tree
                    .update_rp(history_idx, children, strategy, num_player)
            }
        } else if pubnode.node_type == "T" {
        } else {
            panic!("Unknown Node Type")
        }
        (
            crate::dfs::DfsAction::Continue(pubnode.children.clone()),
            None,
        )
    }

    fn on_leaf(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> CfrVal {
        // 更新utility
        let pubnode = &static_ctx.pubtree[idx];
        for &history_idx in pubnode.histories.iter() {
            ctx.utility_tree.update_leaf_utility(
                history_idx,
                &static_ctx.game_state_tree[history_idx]
                    .payoff
                    .as_ref()
                    .unwrap(),
            );
        }
        CfrVal {}
    }

    fn on_pruned(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> CfrVal {
        CfrVal {}
    }

    fn on_accumulate(
        &mut self,
        parent_idx: usize,
        child_idx: usize,
        child_val: CfrVal,
        agg: &mut Option<CfrVal>,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> std::ops::ControlFlow<CfrVal, ()> {
        std::ops::ControlFlow::Continue(())
    }

    fn on_exit(
        &mut self,
        idx: usize,
        agg: Option<CfrVal>,
        _signal: Option<CfrVal>,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DFSExitAction<CfrVal>, CfrVal) {
        // dbg!(format!("pubnode {} on exit", idx));
        let pubnode = &static_ctx.pubtree[idx];
        if pubnode.node_type == "P" {
            // 1. 更新当前节点的utiltiy
            let current_player = static_ctx.get_current_player(idx) as usize;
            let infosets = &pubnode.infosets[current_player];
            let strategy_i = infosets
                .iter()
                .map(|x| ctx.policy_vec[current_player].get_strategy(x))
                .collect::<Vec<_>>();
            for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter()) {
                static_ctx.infoset_dict[current_player][infoset_str]
                    .i2history
                    .iter()
                    .for_each(|&history_idx| {
                        let children = &static_ctx.game_state_tree[history_idx].children;
                        ctx.utility_tree
                            .update_utility(history_idx, children, strategy)
                    });
            }

            for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter()) {
                // 2. 更新平均策略
                let self_rp = ctx.utility_tree.get_self_rp(
                    current_player,
                    static_ctx.infoset_dict[current_player][infoset_str].i2history[0],
                );
                let weight: f64;
                if ctx.cfg.cfr_plus {
                    weight = (ctx.step + 1) as f64 * self_rp;
                } else {
                    weight = self_rp;
                }
                ctx.policy_vec[current_player].update_avg_strategy(infoset_str, weight);
                // 3. 计算当前信息集的cv_ia和cv
                let n_actions = strategy.len();
                let init = vec![0.0f64; n_actions];
                let (cv_ia, cv) = static_ctx.infoset_dict[current_player][infoset_str]
                    .i2history
                    .iter()
                    .map(|&history_idx| {
                        let rp_outside =
                            ctx.utility_tree.get_rp_outside(current_player, history_idx);
                        let cv_ia = static_ctx.game_state_tree[history_idx]
                            .children
                            .iter()
                            .map(|&child_history_idx| {
                                ctx.utility_tree
                                    .get_utility_p(current_player, child_history_idx)
                                    * rp_outside
                            })
                            .collect::<Vec<_>>();
                        let cv = rp_outside
                            * ctx.utility_tree.get_utility_p(current_player, history_idx);
                        (cv_ia, cv)
                    })
                    .fold((init, 0.0), |(mut accum_cv_ia, accum_cv), (cv_ia, cv)| {
                        accum_cv_ia
                            .iter_mut()
                            .zip(cv_ia.iter())
                            .for_each(|(x, y)| *x += y);
                        (accum_cv_ia, accum_cv + cv)
                    });
                // 4. 计算当前信息集的瞬时遗憾，更新Policy
                let regret = cv_ia.into_iter().map(|x| x - cv).collect();
                ctx.policy_vec[current_player].update_regret(
                    infoset_str,
                    &regret,
                    ctx.cfg.cfr_plus,
                );
                // 5. 更新当前策略
                ctx.policy_vec[current_player].update_current_strategy(infoset_str);
            }
        } else if pubnode.node_type == "C" {
            // 1. 更新当前节点的utiltiy
            let histories = &pubnode.histories;
            for &history_idx in histories {
                let strategy = static_ctx.game_state_tree[history_idx]
                    .chance
                    .as_ref()
                    .unwrap();
                let children = &static_ctx.game_state_tree[history_idx].children;
                ctx.utility_tree
                    .update_utility(history_idx, children, strategy)
            }
        } else if pubnode.node_type == "T" {
        } else {
            panic!("Unknown Node Type")
        }
        (crate::dfs::DFSExitAction::Exit, CfrVal {})
    }
}

impl PruneVisitor<DynamicCtx, StaticTree, ParitalPruneVal, CfrVal> for Cfr {
    fn on_enter(
        &mut self,
        idx: usize,
        _depth: usize,
        _signal: &Option<CfrVal>,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DfsAction, Option<CfrVal>) {
        // dbg!(format!("pubnode {} on enter", idx));
        let num_player = ctx.policy_vec.len();
        // 1. 更新子节点的rp
        let pubnode = &static_ctx.pubtree[idx];
        if pubnode.node_type == "P" {
            let current_player = static_ctx.get_current_player(idx) as usize;
            let infosets = &pubnode.infosets[current_player];
            let strategy_i = infosets
                .iter()
                .map(|x| ctx.policy_vec[current_player].get_strategy(x))
                .collect::<Vec<_>>();
            for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter()) {
                static_ctx.infoset_dict[current_player][infoset_str]
                    .i2history
                    .iter()
                    .for_each(|&history_idx| {
                        let children = &static_ctx.game_state_tree[history_idx].children;
                        ctx.utility_tree
                            .update_rp(history_idx, children, strategy, current_player)
                    });
            }
        } else if pubnode.node_type == "C" {
            let histories = &pubnode.histories;
            for &history_idx in histories {
                let strategy = static_ctx.game_state_tree[history_idx]
                    .chance
                    .as_ref()
                    .unwrap();
                let children = &static_ctx.game_state_tree[history_idx].children;
                ctx.utility_tree
                    .update_rp(history_idx, children, strategy, num_player)
            }
        } else if pubnode.node_type == "T" {
        } else {
            panic!("Unknown Node Type")
        }
        (
            crate::dfs::DfsAction::Continue(pubnode.children.clone()),
            None,
        )
    }

    fn on_leaf(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> ParitalPruneVal {
        // 更新utility
        let pubnode = &static_ctx.pubtree[idx];
        for &history_idx in pubnode.histories.iter() {
            ctx.utility_tree.update_leaf_utility(
                history_idx,
                &static_ctx.game_state_tree[history_idx]
                    .payoff
                    .as_ref()
                    .unwrap(),
            );
        }
        ParitalPruneVal {
            num_pruned_states: [0, 0],
            total_states: pubnode.histories.len(),
        }
    }

    fn on_pruned(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> ParitalPruneVal {
        panic!("vanilla cfr does not prune")
    }

    fn on_accumulate(
        &mut self,
        parent_idx: usize,
        child_idx: usize,
        child_val: ParitalPruneVal,
        agg: &mut Option<ParitalPruneVal>,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> std::ops::ControlFlow<ParitalPruneVal, ()> {
        match agg {
            Some(v) => {
                v.num_pruned_states[0] += child_val.num_pruned_states[0];
                v.num_pruned_states[1] += child_val.num_pruned_states[1];
                v.total_states += child_val.total_states;
            }
            None => *agg = Some(child_val),
        }
        std::ops::ControlFlow::Continue(())
    }

    fn on_exit(
        &mut self,
        idx: usize,
        agg: Option<ParitalPruneVal>,
        _signal: Option<CfrVal>,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DFSExitAction<CfrVal>, ParitalPruneVal) {
        // dbg!(format!("pubnode {} on exit", idx));
        let mut num_pruned_states = agg.as_ref().unwrap().num_pruned_states.clone();
        let mut total_states = agg.as_ref().unwrap().total_states.clone();
        let pubnode = &static_ctx.pubtree[idx];
        if pubnode.node_type == "P" {
            // 1. 更新当前节点的utiltiy
            let current_player = static_ctx.get_current_player(idx) as usize;
            let infosets = &pubnode.infosets[current_player];
            let strategy_i = infosets
                .iter()
                .map(|x| ctx.policy_vec[current_player].get_strategy(x))
                .collect::<Vec<_>>();
            for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter()) {
                static_ctx.infoset_dict[current_player][infoset_str]
                    .i2history
                    .iter()
                    .for_each(|&history_idx| {
                        for p in 0..2 {
                            if ctx.utility_tree.get_rp_outside(p, history_idx) == 0.0 {
                                num_pruned_states[p] += 1;
                            }; // 如果当前玩家的外部到达概率是0就不需要计算utility，但是考虑到公共树遍历是为两个玩家同时做的，所以没有当前玩家这一说，真正要裁剪的话需要两个玩家的外部到达概率都为0
                        }
                        total_states += 1;
                        let children = &static_ctx.game_state_tree[history_idx].children;
                        ctx.utility_tree
                            .update_utility(history_idx, children, strategy)
                    });
            }

            for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter()) {
                // 2. 更新平均策略
                let self_rp = ctx.utility_tree.get_self_rp(
                    current_player,
                    static_ctx.infoset_dict[current_player][infoset_str].i2history[0],
                );
                let weight: f64;
                if ctx.cfg.cfr_plus {
                    weight = (ctx.step + 1) as f64 * self_rp;
                } else {
                    weight = self_rp;
                }
                ctx.policy_vec[current_player].update_avg_strategy(infoset_str, weight);
                // 3. 计算当前信息集的cv_ia和cv
                let n_actions = strategy.len();
                let init = vec![0.0f64; n_actions];
                let (cv_ia, cv) = static_ctx.infoset_dict[current_player][infoset_str]
                    .i2history
                    .iter()
                    .map(|&history_idx| {
                        let rp_outside =
                            ctx.utility_tree.get_rp_outside(current_player, history_idx);
                        let cv_ia = static_ctx.game_state_tree[history_idx]
                            .children
                            .iter()
                            .map(|&child_history_idx| {
                                ctx.utility_tree
                                    .get_utility_p(current_player, child_history_idx)
                                    * rp_outside
                            })
                            .collect::<Vec<_>>();
                        let cv = rp_outside
                            * ctx.utility_tree.get_utility_p(current_player, history_idx);
                        (cv_ia, cv)
                    })
                    .fold((init, 0.0), |(mut accum_cv_ia, accum_cv), (cv_ia, cv)| {
                        accum_cv_ia
                            .iter_mut()
                            .zip(cv_ia.iter())
                            .for_each(|(x, y)| *x += y);
                        (accum_cv_ia, accum_cv + cv)
                    });
                // 4. 计算当前信息集的瞬时遗憾，更新Policy
                let regret = cv_ia.into_iter().map(|x| x - cv).collect();
                ctx.policy_vec[current_player].update_regret(
                    infoset_str,
                    &regret,
                    ctx.cfg.cfr_plus,
                );
                // 5. 更新当前策略
                ctx.policy_vec[current_player].update_current_strategy(infoset_str);
            }
        } else if pubnode.node_type == "C" {
            // 1. 更新当前节点的utiltiy
            let histories = &pubnode.histories;
            for &history_idx in histories {
                let strategy = static_ctx.game_state_tree[history_idx]
                    .chance
                    .as_ref()
                    .unwrap();
                let children = &static_ctx.game_state_tree[history_idx].children;
                ctx.utility_tree
                    .update_utility(history_idx, children, strategy);
                for p in 0..2 {
                    if ctx.utility_tree.get_rp_outside(p, history_idx) == 0.0 {
                        num_pruned_states[p] += 1;
                    }; // 如果当前玩家的外部到达概率是0就不需要计算utility，但是考虑到公共树遍历是为两个玩家同时做的，所以没有当前玩家这一说，真正要裁剪的话需要两个玩家的外部到达概率都为0
                }
                total_states += 1;
            }
        } else if pubnode.node_type == "T" {
        } else {
            panic!("Unknown Node Type")
        }
        (
            crate::dfs::DFSExitAction::Exit,
            ParitalPruneVal {
                num_pruned_states: num_pruned_states,
                total_states: total_states,
            },
        )
    }
}

pub fn cfr(game_name: &str, mut logger: Logger, cfg: crate::Config) {
    println!("loading {}", game_name);
    let static_trees: (
        Vec<PubNode>,
        Vec<Vec<InfoSet>>,
        Vec<HashMap<String, InfoSet>>,
        Vec<History>,
    ) = from_json(&format!("tree/{}/trees_{}_0923.txt", game_name, game_name));
    let policy_vec = build_policy(&static_trees);
    let utility_tree = build_utiltiy_tree(&static_trees);
    let mut ctx = DynamicCtx {
        step: 0,
        cfg,
        policy_vec,
        utility_tree,
    };
    let static_ctx = StaticTree {
        pubtree: static_trees.0,
        infoset_dict: static_trees.2,
        game_state_tree: static_trees.3,
    };
    let num_player = static_ctx.infoset_dict.len();
    let mut timer = CfrTimerWithSamples::new(50_000, 42); // 最多保留5万样本
    let bar = ProgressBar::new(ctx.cfg.epoch.try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            // {eta} 就是预计完成时间；{elapsed} 已耗时；{per_sec} 速度
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} \
             ({percent}%) • {per_sec} it/s • ETA {eta}",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    let mut num_not_pruned_node = 0;
    let mut pp_prune = [0, 0];
    let mut total_num_game_states = 0;
    for step in 0..ctx.cfg.epoch {
        // if let Err(e) = record_policy_to_csv(&mut ctx, step, "policy_log.csv") {
        //     eprintln!("Failed to record policy at step {}: {:?}", step, e);
        // }
        bar.inc(1);
        let mut m = BTreeMap::new();
        let t0 = Instant::now();
        let pp = dfs_with_pruning_backprop::<DynamicCtx, StaticTree, ParitalPruneVal, CfrVal, Cfr>(
            0,
            &mut ctx,
            &static_ctx,
            &mut Cfr {},
        );
        if let Some(pp) = pp {
            pp_prune
                .iter_mut()
                .zip(pp.num_pruned_states.iter())
                .for_each(|(x, y)| *x += y);
            total_num_game_states += pp.total_states;
        }
        let dt = t0.elapsed();
        timer.add(dt, 1);
        ctx.step += 1;
        num_not_pruned_node += static_ctx.pubtree.len();
        if ctx.cfg.record_step.contains(&step) {
            for player in 0..num_player {
                m.insert(
                    format!("current_return_{}", player),
                    ctx.get_root_value(player),
                );
            }
            let mut ctx = br::DynamicCtx {
                br_player: 99, // 设为99是为了计算二者在平均策略下的收益
                policy_vec: &mut ctx.policy_vec,
                utility_tree: &mut ctx.utility_tree,
            };
            dfs_with_pruning_backprop(0, &mut ctx, &static_ctx, &mut br::BestResponse {});
            let avg_value = (0..num_player)
                .into_iter()
                .map(|x| ctx.get_root_value(x))
                .collect::<Vec<_>>();
            let mut best_reponse_value = vec![];
            for br_player in 0..num_player {
                ctx.br_player = br_player;
                dfs_with_pruning_backprop(0, &mut ctx, &static_ctx, &mut br::BestResponse {});
                best_reponse_value.push(ctx.get_root_value(br_player));
            }
            // dbg!(&best_reponse_value);
            // dbg!(&avg_value);
            let exploit = best_reponse_value
                .iter()
                .zip(avg_value.iter())
                .map(|(x, y)| x - y)
                .sum::<f64>()
                / num_player as f64;
            // dbg!(exploit);

            m.insert("exploit".to_string(), exploit);
            for p in 0..num_player {
                m.insert(format!("avg_return_{}", p), avg_value[p]);
            }
            for p in 0..num_player {
                m.insert(format!("br_return_{}", p), best_reponse_value[p]);
            }
            m.insert("avg_time(ms)".to_string(), timer.average_ms());
            m.insert(
                "num_not_pruned_node".to_string(),
                num_not_pruned_node as f64,
            );
            m.insert("num_game_states".to_owned(), total_num_game_states as f64);
            m.insert("pp_game_states1".to_owned(), pp_prune[0] as f64);
            m.insert("pp_game_states2".to_owned(), pp_prune[1] as f64);
            logger.log_step(step, &m).unwrap();
            num_not_pruned_node = 0;
            pp_prune = [0, 0];
            total_num_game_states = 0;
        }
    }
}

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

fn record_policy_to_csv(ctx: &mut DynamicCtx, step: usize, csv_path: &str) -> std::io::Result<()> {
    let info_key_p1 = "0_0_P1|public:none|private:about_to_move".to_string();
    let info_key_p2 = "1_1_P2|public:none|private:about_to_move_unknown_P1".to_string();

    let strategy_p1 = ctx.policy_vec[0].get_strategy(&info_key_p1);
    let strategy_p2 = ctx.policy_vec[1].get_strategy(&info_key_p2);

    let (p1_a0, p1_a1) = (strategy_p1[0], strategy_p1[1]);
    let (p2_b0, p2_b1) = (strategy_p2[0], strategy_p2[1]);

    // 若文件不存在则写入表头
    let file_exists = Path::new(csv_path).exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)?;

    if !file_exists {
        writeln!(file, "step,p1_A0,p1_A1,p2_B0,p2_B1")?;
    }

    // 写入一行策略
    writeln!(
        file,
        "{},{:.6},{:.6},{:.6},{:.6}",
        step, p1_a0, p1_a1, p2_b0, p2_b1
    )?;
    Ok(())
}
