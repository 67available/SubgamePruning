use std::ops::ControlFlow;

pub enum DfsAction {
    /// 剪掉该节点整棵子树（不压栈），随后会调用 on_pruned 产出一个回溯值给父节点
    PruneSubtree,
    /// 继续下潜；提供需要被遍历的子节点列表
    Continue(Vec<usize>),
    /// 立即终止整个搜索（顶层返回）
    StopAll,
}

pub enum DFSExitAction<T> {
    /// restart重新将当前节点压入（跳过on_enter）
    Restart(T),
    /// 正常退出
    Exit,
}

/// 访问器：定义剪枝决策、叶子/剪枝的回溯值、父节点如何累积子结果，以及节点退出时如何给出本节点的最终值。
pub trait PruneVisitor<Ctx, StaticCtx, ValUp, ValDown> {
    /// 进入节点时的决策（是否剪枝、如何排序孩子）
    fn on_enter(
        &mut self,
        idx: usize,
        depth: usize,
        signal: &Option<ValDown>, // 来自父节点的frame
        ctx: &mut Ctx,
        static_ctx: &StaticCtx,
    ) -> (DfsAction, Option<ValDown>);

    /// 告诉框架：这是叶子。返回要回溯给父节点的值。
    fn on_leaf(&mut self, idx: usize, depth: usize, ctx: &mut Ctx, static_ctx: &StaticCtx)
    -> ValUp;

    /// 告诉框架：这是被剪枝的节点（未展开）。返回要回溯给父节点的值。
    fn on_pruned(
        &mut self,
        idx: usize,
        depth: usize,
        ctx: &mut Ctx,
        static_ctx: &StaticCtx,
    ) -> ValUp;

    /// 父节点如何用“一个子结果”来更新自己的聚合状态。
    ///
    /// - `agg` 是当前已聚合的父节点状态（你的“中间值”）；`val` 是刚刚完成的这个孩子的值。
    /// - 返回 `Continue(())` 继续处理下一个孩子；返回 `Break(final_parent_val)` 表示可以**提前截断兄弟**，
    ///   并把 `final_parent_val` 作为父节点的最终聚合值进入 `on_exit`。
    fn on_accumulate(
        &mut self,
        parent_idx: usize,
        child_idx: usize,
        child_val: ValUp,
        agg: &mut Option<ValUp>,
        depth: usize,
        ctx: &mut Ctx,
        static_ctx: &StaticCtx,
    ) -> ControlFlow<ValUp, ()>;

    /// 所有（或被截断的）子处理完毕后，得到“本节点”的最终值，回溯给上层。
    /// 参数 `agg` 为累计好的父节点中间值（若没有子或都被剪，可能是 None）。
    fn on_exit(
        &mut self,
        idx: usize,
        agg: Option<ValUp>,
        signal: Option<ValDown>,
        depth: usize,
        ctx: &mut Ctx,
        static_ctx: &StaticCtx,
    ) -> (DFSExitAction<ValDown>, ValUp);
}

// 每一个 Frame 与一个 PubNode 一一对应，负责保存它在 DFS 中的当前处理状态。
#[derive(Debug, Clone)]
struct Frame<ValUp: Clone, ValDown: Clone> {
    idx: usize,
    depth: usize,
    children: Vec<usize>,
    next: usize,
    agg: Option<ValUp>,      // 父节点对已完成子结果的聚合中间值
    signal: Option<ValDown>, // 父节点向子节点传递信息
}

impl<ValUp: Clone, ValDown: Clone> Frame<ValUp, ValDown> {
    fn reset(&mut self, signal: Option<ValDown>) {
        self.next = 0;
        self.agg = None;
        self.signal = signal;
    }
}

/// 为了最好地体现剪枝的计算效率提升，串行执行树遍历
pub fn dfs_with_pruning_backprop<Ctx, StaticCtx, ValUp: Clone, ValDown: Clone, V>(
    root: usize,
    ctx: &mut Ctx,          // 动态上下文信息
    static_ctx: &StaticCtx, // 静态上下文信息
    visitor: &mut V,
) -> Option<ValUp>
where
    V: PruneVisitor<Ctx, StaticCtx, ValUp, ValDown>,
{
    // 进入根：若剪枝/停止则直接给出值
    let (initial_action, val_down) =
        visitor.on_enter(root, 0, &Option::<ValDown>::None, ctx, static_ctx);
    let mut stack: Vec<Frame<ValUp, ValDown>> = Vec::with_capacity(64);

    match initial_action {
        DfsAction::StopAll => return None,
        DfsAction::PruneSubtree => {
            // 根被剪 -> 调用 on_pruned，再结束
            let v = visitor.on_pruned(root, 0, ctx, static_ctx);
            return Some(v);
        }
        DfsAction::Continue(ch) => {
            stack.push(Frame {
                idx: root,
                depth: 0,
                children: ch,
                next: 0, // 待处理的 children 的 idx
                agg: None,
                signal: val_down,
            });
        }
    }

    // 用一个局部变量暂存“刚完成的子结果”，以便喂给父帧的 on_accumulate
    let mut carry_child_val: Option<ValUp> = None;

    loop {
        let Some(mut frame) = stack.pop() else {
            // 栈空：根节点刚刚完成；其 on_exit 的值存在 carry_child_val 里
            return carry_child_val;
        };

        // 若有一个子结果刚从下层回来，应该先把它累计到当前父节点 frame 上
        if let Some(child_val) = carry_child_val.take() {
            let parent_idx = frame.idx;
            let child_idx = frame.children[frame.next - 1]; // 刚处理完的那个孩子
            match visitor.on_accumulate(
                parent_idx,
                child_idx,
                child_val,
                &mut frame.agg,
                frame.depth,
                ctx,
                static_ctx,
            ) {
                ControlFlow::Continue(()) => {
                    // 继续处理下一个孩子（下面会统一处理）
                }
                ControlFlow::Break(parent_final) => {
                    // 不再处理该父的其他孩子，直接跳到on_exit
                    frame.next = frame.children.len();
                }
            }
        }

        // 还有没处理的孩子？
        if frame.next < frame.children.len() {
            let v = frame.children[frame.next]; // 这里的frame是父节点的frame，而v是子节点的idx
            frame.next += 1;

            // 子节点进入决策
            let child_depth = frame.depth + 1;
            let (dfs_action, val_down) =
                visitor.on_enter(v, child_depth, &frame.signal, ctx, static_ctx);

            stack.push(frame); // 把父帧先放回去（等待子完成后继续聚合）
            match dfs_action {
                DfsAction::StopAll => {
                    // 顶层“停止全部”：直接返回当前已有根值（若需要更复杂语义可自定义）
                    return None;
                }
                DfsAction::PruneSubtree => {
                    // 子被剪：立刻得到该子要回传给父的值，塞进 carry_child_val，下一轮聚合
                    let vval = visitor.on_pruned(v, child_depth, ctx, static_ctx);
                    carry_child_val = Some(vval);
                    continue;
                }
                DfsAction::Continue(ch2) => {
                    // 叶节点直接调用 on_leaf，作为一个“子结果”回传给父。注意：叶节点不会触发on_exit!
                    if ch2.is_empty() {
                        let leafv = visitor.on_leaf(v, child_depth, ctx, static_ctx);
                        carry_child_val = Some(leafv);
                        continue;
                    } else {
                        // 真正展开子 -> 压栈
                        stack.push(Frame {
                            idx: v,
                            depth: child_depth,
                            children: ch2,
                            next: 0,
                            agg: None,
                            signal: val_down,
                        });
                        continue;
                    }
                }
            }
        } else {
            // 没有孩子或都处理完/被截断 -> 本节点退出，产出本节点值，回传给父
            let mut bkp_frame = frame.clone();
            let v = visitor.on_exit(
                frame.idx,
                frame.agg,
                frame.signal,
                frame.depth,
                ctx,
                static_ctx,
            );
            match v.0 {
                DFSExitAction::Restart(singal) => {
                    bkp_frame.reset(Some(singal));
                    stack.push(bkp_frame); // 把父帧放回去
                }
                DFSExitAction::Exit => carry_child_val = Some(v.1),
            }
            // 继续上一层
            continue;
        }
    }
}
