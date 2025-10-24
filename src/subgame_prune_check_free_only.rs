use crate::{
    DELTA_RP, JUMPTHEGUN, WARMUP,
    br::{self, argmax},
    cfr::{Policy, StaticTree, UtilityTree, build_policy, build_utiltiy_tree},
    dfs::{PruneVisitor, dfs_with_pruning_backprop},
    load_tree::{History, InfoSet, PubNode, from_json},
    logger::Logger,
    timer::CfrTimerWithSamples,
};
use core::f64;
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};

/// 存储用与剪枝判定与RM补偿相关的信息，
struct PruneInfo {
    zero_regret: Vec<bool>,
    duration: Vec<usize>, // 在alpha根节点记录被裁剪了多少次迭代
}

struct DynamicCtx {
    pub cfg: crate::Config,
    pub step: usize,
    pub policy_vec: Vec<Policy>,
    pub utility_tree: UtilityTree,
    pub prune_info: PruneInfo,
}

impl DynamicCtx {
    pub fn get_root_value(&self, player: usize) -> f64 {
        self.utility_tree.get_utility_p(player, 0)
    }
}

#[derive(Clone)]
struct SubgamePruneVal {
    max_regret: f64,
    num_not_pruned_node: usize,
}

#[derive(Clone, Debug)]
struct CompensateVal {
    root_of_pruned_subgame: usize,
}

struct SubgamePruneCheckFreeOnly;
/// 先实现一般只考虑CheckFree情形的代码，理论上只考虑CheckFree的话计算效率肯定会比vanillaCfr来的高
/// 这种情况下，Prune Constraint为 a.子博弈中所有信息集的最大regret==0 ; b.子博弈根节点的rp不变
impl PruneVisitor<DynamicCtx, StaticTree, SubgamePruneVal, CompensateVal>
    for SubgamePruneCheckFreeOnly
{
    fn on_enter(
        &mut self,
        idx: usize,
        depth: usize,
        signal: &Option<CompensateVal>,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DfsAction, Option<CompensateVal>) {
        let num_player = ctx.policy_vec.len();
        let pubnode = &static_ctx.pubtree[idx];
        let mut compensate = signal.is_some(); // 如果singal不为None，说明这个节点是某个待补偿子博弈的非根节点        
        let mut root_of_compensate = false;
        // 1. （对于被裁剪子博弈的根节点）判断节点是否为满足裁剪条件 a. 上一次更新max_regret时，max_regret -> 0 b. 当前rp与上一次更新时的rp一致
        if ctx.prune_info.zero_regret[idx]
            // && ctx.prune_info.zero_regret_c[idx] > JUMPTHEGUN
            && !compensate
        {
            // 1.1 检查当前rp与上一次更新时的rp是否一致 / 接近
            let infosets = &pubnode.infosets;
            let mut rp_diff = 0.0;
            for (p, infosets_p) in infosets.iter().enumerate() {
                let tmp_infoset_dict = &static_ctx.infoset_dict[p];
                for infoset_str in infosets_p {
                    let history_idx = tmp_infoset_dict[infoset_str].i2history[0];
                    rp_diff += ctx.utility_tree.get_self_rp_diffence(p, history_idx);
                }
            }

            let histories = &pubnode.histories;
            // 1.2 对当前节点内所有的历史做rp累积
            for &history_idx in histories {
                ctx.utility_tree.accum_rp(history_idx);
            }
            // 1.3 记录当次裁剪持续的时间
            ctx.prune_info.duration[idx] += 1;
            // 1.4 判断是否进行RM补偿

            if rp_diff <= DELTA_RP && !ctx.cfg.force_compensate {
                // dbg!("check-free", idx);
                // 1.4.1 如果一致则直接退出
                return (crate::dfs::DfsAction::PruneSubtree, None); // 在CheckFreeOnly情况下，如果被裁剪该节点就不用入栈了
            } else {
                // dbg!("compensate", idx);
                // 1.4.2 如果不一致则开启RM补偿（补偿平均策略与累积遗憾值）
                root_of_compensate = true;
                compensate = true;
            }
        }
        // 2. 更新子节点的last_rp，由于在更新子节点rp时会直接把last_rp覆盖掉，所以如果子节点被裁剪的话，需要提前把子节点的last_rp存储起来。另一种实现方式是我每次更新rp的时候都存一下last_rp，如果被裁剪则不更新last_rp。
        // 选择第二种实现方式，这部分代码挪到3. 中
        // 3. 更新子节点的rp
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
                            .update_rp(history_idx, children, strategy, current_player);
                        //  ------ subgame_prune  -------//
                        ctx.utility_tree.update_last_rp(history_idx);
                        if compensate {
                            // if root_of_compensate {
                            //     ctx.utility_tree
                            //         .set_alpha_history_idx(history_idx, history_idx);
                            //     ctx.utility_tree
                            //         .set_alpha2h_rp(history_idx, vec![1.0; num_player + 1]);
                            // }
                            // ctx.utility_tree
                            //     .update_alpha_history_idx(history_idx, &children); // 对于CheckFreeOnly只需要在compensate的那次迭代把alpha_history_idx传下去即可
                            // ctx.utility_tree.update_alpha2h_rp(
                            //     history_idx,
                            //     &children,
                            //     strategy,
                            //     current_player,
                            // );
                            ctx.utility_tree.update_all(
                                history_idx,
                                &children,
                                strategy,
                                current_player,
                                num_player,
                                root_of_compensate,
                            );
                        }
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
                    .update_rp(history_idx, children, strategy, num_player);
                //  ------ subgame_prune  -------//
                ctx.utility_tree.update_last_rp(history_idx);
                if compensate {
                    // if root_of_compensate {
                    //     ctx.utility_tree
                    //         .set_alpha_history_idx(history_idx, history_idx);
                    //     ctx.utility_tree
                    //         .set_alpha2h_rp(history_idx, vec![1.0; num_player + 1]);
                    // }
                    // ctx.utility_tree
                    //     .update_alpha_history_idx(history_idx, &children); // 对于CheckFreeOnly只需要在compensate的那次迭代把alpha_history_idx传下去即可
                    // // dbg!(history_idx, &children);
                    // ctx.utility_tree.update_alpha2h_rp(
                    //     history_idx,
                    //     &children,
                    //     strategy,
                    //     num_player,
                    // );
                    // // dbg!("update_alpha_rp_chance", idx, history_idx, histories);
                    ctx.utility_tree.update_all(
                        history_idx,
                        &children,
                        strategy,
                        num_player,
                        num_player,
                        root_of_compensate,
                    );
                }
            }
        } else if pubnode.node_type == "T" {
        } else {
            panic!("Unknown Node Type")
        }
        let val_down;
        if root_of_compensate {
            val_down = Some(CompensateVal {
                root_of_pruned_subgame: idx,
            });
        } else {
            val_down = signal.clone();
        }
        (
            crate::dfs::DfsAction::Continue(pubnode.children.clone()),
            val_down,
        )
    }

    fn on_leaf(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> SubgamePruneVal {
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
        SubgamePruneVal {
            max_regret: 0.0,
            num_not_pruned_node: 1,
        }
    }

    fn on_pruned(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> SubgamePruneVal {
        SubgamePruneVal {
            max_regret: 99.0,
            num_not_pruned_node: 0,
        } // 之所以置为1.0是为了避免嵌套裁剪，如果要考虑嵌套裁剪的话就需要在每次开启裁剪时对子博弈内的所有正则裁剪的子子博弈进行中止、RM补偿。
    }

    fn on_accumulate(
        &mut self,
        parent_idx: usize,
        child_idx: usize,
        child_val: SubgamePruneVal,
        agg: &mut Option<SubgamePruneVal>,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> std::ops::ControlFlow<SubgamePruneVal, ()> {
        match agg {
            Some(v) => {
                v.max_regret = v.max_regret.max(child_val.max_regret);
                v.num_not_pruned_node += child_val.num_not_pruned_node;
            }
            None => *agg = Some(child_val),
        }
        std::ops::ControlFlow::Continue(())
    }

    fn on_exit(
        &mut self,
        idx: usize,
        agg: Option<SubgamePruneVal>,
        signal: Option<CompensateVal>,
        depth: usize,
        ctx: &mut DynamicCtx,
        static_ctx: &StaticTree,
    ) -> (crate::dfs::DFSExitAction<CompensateVal>, SubgamePruneVal) {
        // max_reget
        let mut max_regret = if let Some(v) = agg.as_ref() {
            v.max_regret
        } else {
            0.0
        };
        let pubnode = &static_ctx.pubtree[idx];
        let compensate = signal.is_some(); // 如果singal不为None，则这个节点需要做补偿，如果这个节点又恰好是被裁减子博弈的根节点
        if pubnode.node_type == "P" {
            let current_player = static_ctx.get_current_player(idx) as usize;
            let infosets = &pubnode.infosets[current_player];
            let strategy_i = infosets
                .iter()
                .map(|x| ctx.policy_vec[current_player].get_strategy(x))
                .collect::<Vec<_>>();
            // 1. 更新当前节点的utiltiy
            // 在compensate那一次迭代不需要更新utility
            if !compensate {
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
            }
            for (infoset_str, strategy) in infosets.iter().zip(strategy_i.iter()) {
                // 2. 更新平均策略
                let self_rp = if compensate {
                    // RM补偿
                    ctx.utility_tree.get_self_rp_compensate(
                        current_player,
                        static_ctx.infoset_dict[current_player][infoset_str].i2history[0],
                    )
                } else {
                    ctx.utility_tree.get_self_rp(
                        current_player,
                        static_ctx.infoset_dict[current_player][infoset_str].i2history[0],
                    )
                };
                let weight: f64;
                if ctx.cfg.cfr_plus {
                    weight = if compensate {
                        let root_of_prune_subgame = signal.as_ref().unwrap().root_of_pruned_subgame;
                        let ti = ((ctx.step + 1)
                            + (ctx.step - ctx.prune_info.duration[root_of_prune_subgame] + 2)) as f64
                            // * ctx.prune_info.duration[root_of_prune_subgame] 
                            / 2.0;
                        ti * self_rp // 这里计算的并不是精确的weight而是一个近似，如果想要精确计算weight，需要在累加rp的时候就对每个step独立加权
                    } else {
                        (ctx.step + 1) as f64 * self_rp
                    };
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
                        // RM补偿，此处可以直接修改cv_ia为累积cv_ia，但是要注意不能直接用这个累积cv_ia计算max_regret，max_regret需要用当前regret计算，对于cfr_plus而言这里不需要做补偿，所以对于cfr而言我们其实也可以不补偿，这样在经验上可能会有比cfr更快的收敛效果，但是如果要保持与cfr的一致性的话还是得补偿。
                        let rp_outside = if compensate {
                            ctx.utility_tree
                                .get_rp_outside_compensate(current_player, history_idx)
                        } else {
                            ctx.utility_tree.get_rp_outside(current_player, history_idx)
                        };

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
                // 4.1 计算当前信息集的瞬时遗憾，更新Policy
                let regret = cv_ia.into_iter().map(|x| x - cv).collect::<Vec<_>>();
                ctx.policy_vec[current_player].update_regret(
                    infoset_str,
                    &regret,
                    ctx.cfg.cfr_plus,
                );
                // 4.2 计算最大遗憾值，用与PruneConstraint判定
                // 如果前头做了RM补偿，则regret为累积regret而不是当前的regret，当前的regret还得重新算，一种方式是认为当一个子博弈刚被补偿后它不可能马上又被裁剪，所以我直接将max_regret设为一个足够大的数
                if compensate {
                    max_regret = 99.0
                } else {
                    let max_regret_idx = argmax(&regret).unwrap();
                    if regret[max_regret_idx] > max_regret {
                        max_regret = regret[max_regret_idx]
                    }
                }
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
        // 如果此时max_regret == 0 ， 说明以这个节点为根节点的子博弈是可以裁剪的
        if max_regret <= crate::DELTA_REGRET {
            ctx.prune_info.zero_regret[idx] = true;
            // ctx.prune_info.zero_regret_c[idx] += 1;
        } else {
            ctx.prune_info.zero_regret[idx] = false;
            // ctx.prune_info.zero_regret_c[idx] = 0;
        }
        if compensate && signal.unwrap().root_of_pruned_subgame == idx {
            // dbg!("clean accum rp ", idx);
            // 如果进行了补偿且该节点时被裁剪博弈的根节点，则需要清空accum_rp
            pubnode
                .histories
                .iter()
                .for_each(|&x| ctx.utility_tree.reset(x)); // 清空accum_rp
            ctx.prune_info.duration[idx] = 0;
        }
        (
            crate::dfs::DFSExitAction::Exit,
            SubgamePruneVal {
                max_regret,
                num_not_pruned_node: if compensate {
                    // 是否计入compensate阶段遍历的节点，计入会比较公平一点，因为compensate的计算量与正常cfr迭代相当，不能忽略
                    1 + agg.as_ref().unwrap().num_not_pruned_node // 0
                } else {
                    1 + agg.as_ref().unwrap().num_not_pruned_node
                },
            },
        )
    }
}

pub fn subgame_prune_cfr(game_name: &str, mut logger: Logger, cfg: crate::Config) {
    println!("loading {}", game_name);
    let static_trees: (
        Vec<PubNode>,
        Vec<Vec<InfoSet>>,
        Vec<HashMap<String, InfoSet>>,
        Vec<History>,
    ) = from_json(&format!("tree/{}/trees_{}_0923.txt", game_name, game_name));
    let policy_vec = build_policy(&static_trees);
    let utility_tree = build_utiltiy_tree(&static_trees);
    let mut utility_tree4br = build_utiltiy_tree(&static_trees); // 计算exploit用的utility_tree，避免计算exploit时把subgame prune里需要存储的信息覆盖了
    let prune_info = PruneInfo {
        zero_regret: vec![false; static_trees.3.len()],
        duration: vec![0; static_trees.3.len()],
    };
    let mut ctx = DynamicCtx {
        step: 0,
        cfg,
        policy_vec,
        utility_tree,
        prune_info,
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
    for step in 0..ctx.cfg.epoch {
        bar.inc(1);
        if ctx.cfg.record_step.contains(&step) {
            ctx.cfg.force_compensate = true;
        } else {
            ctx.cfg.force_compensate = false;
        }
        let mut m = BTreeMap::new();
        let t0 = Instant::now();
        let val =
            dfs_with_pruning_backprop(0, &mut ctx, &static_ctx, &mut SubgamePruneCheckFreeOnly {});
        let dt = t0.elapsed();
        if step > WARMUP {
            timer.add(dt, 1);
        }
        ctx.step += 1;
        num_not_pruned_node += val.as_ref().unwrap().num_not_pruned_node;
        if ctx.cfg.record_step.contains(&step) {
            for player in 0..num_player {
                m.insert(
                    format!("current_return_{}", player),
                    ctx.get_root_value(player),
                );
            }

            let mut ctx = br::DynamicCtx {
                br_player: 99, // 设为99是为了计算二者在平均策略下的收益
                policy_vec: &ctx.policy_vec,
                utility_tree: &mut utility_tree4br,
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
            let exploit = best_reponse_value
                .iter()
                .zip(avg_value.iter())
                .map(|(x, y)| x - y)
                .sum::<f64>()
                / num_player as f64;

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
            m.insert(
                "max_regret".to_string(),
                val.as_ref().unwrap().max_regret as f64,
            );
            logger.log_step(step, &m).unwrap();
            num_not_pruned_node = 0;
        }
    }
}
