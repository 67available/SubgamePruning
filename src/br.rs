use crate::{
    cfr::{Policy, StaticTree, UtilityTree},
    dfs::PruneVisitor,
};
use std::cmp::Ordering;

/// 返回切片中最大元素的下标；若为空或没有有限值则返回 None
pub fn argmax(v: &[f64]) -> Option<usize> {
    v.iter()
        .enumerate()
        // 忽略 NaN / ±∞（如需保留可去掉这一行）
        .filter(|(_, x)| x.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
}

/// 动态上下文 Policy与Utility Tree
pub struct DynamicCtx<'a> {
    pub br_player: usize,
    pub policy_vec: &'a Vec<Policy>,
    pub utility_tree: &'a mut UtilityTree,
}

impl<'a> DynamicCtx<'a> {
    /// 获取br_player在面对对手平均策略时的最优响应
    pub fn get_root_value(&self, player: usize) -> f64 {
        self.utility_tree.get_utility_p(player, 0)
    }
}

#[derive(Clone)]
pub struct BrVal {}

pub struct BestResponse;

impl<'a> PruneVisitor<DynamicCtx<'a>, StaticTree, BrVal, BrVal> for BestResponse {
    fn on_enter(
        &mut self,
        idx: usize,
        _depth: usize,
        _signal: &Option<BrVal>,
        ctx: &mut DynamicCtx<'a>,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DfsAction, Option<BrVal>) {
        let num_player = ctx.policy_vec.len();
        // 1. 使用平均策略更新子节点的rp
        let pubnode = &static_ctx.pubtree[idx];
        if pubnode.node_type == "P" {
            let current_player = static_ctx.get_current_player(idx) as usize;
            let infosets = &pubnode.infosets[current_player];
            let strategy_i = infosets
                .iter()
                .map(|x| ctx.policy_vec[current_player].get_avg_strategy(x))
                .collect::<Vec<_>>(); // 与VallinaCfr的on_enter只差在这一行，其他地方一致
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
        ctx: &mut DynamicCtx<'a>,
        static_ctx: &StaticTree,
    ) -> BrVal {
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
        BrVal {}
    }

    fn on_pruned(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut DynamicCtx<'a>,
        static_ctx: &StaticTree,
    ) -> BrVal {
        BrVal {}
    }

    fn on_accumulate(
        &mut self,
        parent_idx: usize,
        child_idx: usize,
        child_val: BrVal,
        agg: &mut Option<BrVal>,
        depth: usize,
        ctx: &mut DynamicCtx<'a>,
        static_ctx: &StaticTree,
    ) -> std::ops::ControlFlow<BrVal, ()> {
        std::ops::ControlFlow::Continue(())
    }

    fn on_exit(
        &mut self,
        idx: usize,
        agg: Option<BrVal>,
        _signal: Option<BrVal>,
        depth: usize,
        ctx: &mut DynamicCtx<'a>,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DFSExitAction<BrVal>, BrVal) {
        let pubnode = &static_ctx.pubtree[idx];
        if pubnode.node_type == "P" {
            // 1. 获取 BR v.s. avg 策略
            let current_player = static_ctx.get_current_player(idx) as usize;
            let infosets = &pubnode.infosets[current_player];
            let mut strategy_i = infosets
                .iter()
                .map(|x| ctx.policy_vec[current_player].get_avg_strategy(x))
                .collect::<Vec<_>>();
            if current_player == ctx.br_player {
                // 如果当前玩家是BR玩家需要将平均策略替换成BR策略（如果ctx.br_player>=num_player则相当于计算avg策略下的玩家收益）
                for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter_mut()) {
                    // 1.1 计算当前信息集的cv_ia和cv
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
                    // 1.2 构造one-hot最佳响应
                    let best_action_idx = argmax(&cv_ia).unwrap();
                    strategy.iter_mut().enumerate().for_each(|(idx, x)| {
                        if idx == best_action_idx {
                            *x = 1.0
                        } else {
                            *x = 0.0
                        }
                    });
                }
            }
            // 2. 使用最佳响应/平均策略更新当前节点的utility
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
        (crate::dfs::DFSExitAction::Exit, BrVal {})
    }
}
