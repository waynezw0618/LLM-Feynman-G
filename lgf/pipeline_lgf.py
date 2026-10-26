"""lgf.pipeline_lgf — 分层双循环主控 (LGF 主算法).

外层: LLM 语义级, 每 K 代一次干预 (T_PROP / H_DECOMP / M_GUIDE / E_CHECK).
内层: GP 图进化, 每代一次 (crossover / mutation / selection).
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import LGFConfig
from .graph import Graph
from .priors import run_priors, profile_summary
from .templates import TEMPLATE_REGISTRY, list_templates, build as templates_build
from . import gp as gp_ops
from . import llm_ops
from . import attribution
from .paramfit import fit_joint, evaluate
from .benchmarks import Benchmark, REGISTRY, make_cases


# ============================================================================
# 结果 / 日志结构
# ============================================================================
@dataclass
class GenLog:
    gen: int
    best_loss: float
    best_graph: str
    n_pop: int
    seconds: float
    note: str = ""


def _force_global_params(g: Graph) -> bool:
    """true-ASM (2026-10-09): 把全部逐工况参数 K# 改写为全局 C#.

    策略: 保留 LLM 候选的结构, 强制 ASM 普适性 (常数跨工况共享) —
    优于直接拒绝 (few-shot 范例含 K1, LLM 模仿输出 K 会浪费 G_INIT 名额).
    只在新解析/深拷贝的图上调用 (Node.clone 深拷贝 Param, 不污染父图);
    改写后重跑 _normalize_params 把名字 K#→C# 规范化.
    返回是否发生了改写.
    """
    from .graph import walk
    flipped = False
    for _, node in walk(g.root):
        for _, p in node.children:
            if p is not None and p.kind == "case":
                p.kind = "global"
                flipped = True
    if flipped:
        g._normalize_params()
    return flipped


def run_lgf(cases: List[dict], var_names: Sequence[str], cfg: LGFConfig,
            log_dir: str, llm_client, profile: Optional[Dict] = None,
            prior_constraints=None, name: str = "run") -> Dict:
    """单次 LGF 运行主入口."""
    os.makedirs(log_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    templates: List[str] = list(TEMPLATE_REGISTRY.keys())
    log_path = os.path.join(log_dir, f"{name}.jsonl")
    genlog = []
    prior_report: Optional[Dict] = None

    # V3-a (2026-10-08): LLM 通道调用统计 — 落盘进 final.json.
    # 背景: P1c/P1d 全部 run 的 M_GUIDE 因 Graph.clone 不拷贝 theta 而静默失效,
    # 除 n_pop 恒定外无任何痕迹, 改进是否生效完全不可观测.
    llm_stats: Dict = {
        "ginit": {"n_candidates": 0, "n_parsed": 0},
        "mguide": {"n_calls": 0, "n_nonempty": 0, "n_applied": 0, "n_alive": 0,
                   "last_error": None, "last_exception": None},
        "forbidden_filtered": 0,
    }
    FORBIDDEN: set = set(getattr(cfg, "forbidden_ops", None) or [])

    # 1) 数据画像 (+ 可选: AI-Feynman 先验检验)
    if profile is None:
        profile = _quick_profile(cases, var_names)
    if cfg.use_priors:
        try:
            prior_report, prior_constraints = run_priors(cases, var_names)
        except Exception:
            prior_report, prior_constraints = None, None
    else:
        prior_constraints = None

    # 2) 初始化种群: LLM G_INIT + 先验注入 + 模板注入 + 随机图
    pop: List[Graph] = []
    if prior_constraints is not None:
        # Feynman 先验 → 结构化初始图 (可分性 → 对应模板骨架)
        try:
            for (i, j) in prior_constraints.multiplicative_partitions:
                if len(var_names) >= 2:
                    g, _ = templates_build("mul2", int(i), int(j))
                    pop.append(Graph(g.root.clone(), list(var_names), name="prior_mul"))
            for (i, j) in prior_constraints.additive_partitions:
                g, _ = templates_build("lin", int(i))
                pop.append(Graph(g.root.clone(), list(var_names), name="prior_add"))
            pop = pop[: max(2, cfg.population // 4)]
        except Exception:
            pop = []
    # v3a-A1+A3: G_INIT 鲁棒性调用 (解析重试 + few-shot 门控)
    # true-ASM: cfg.physics_context 注入 G_INIT prompt; cfg.global_params_only
    # 拒绝含逐工况常数 K# 的候选 (真 ASM 只允许全局常数, 2026-10-09)
    _phys_ctx: str = getattr(cfg, "physics_context", "") or ""
    _gpo: bool = bool(getattr(cfg, "global_params_only", False))
    init_dsls: List[str] = []
    if cfg.use_llm:
        n_target = cfg.population // 2
        fewshot_on = bool(getattr(cfg, "ginit_fewshot", True))
        max_retry = max(0, int(getattr(cfg, "ginit_retry", 2)))
        min_kept = max(1, int(getattr(cfg, "ginit_min_kept", 3)))
        for _attempt in range(max_retry + 1):
            batch = llm_ops.G_INIT(llm_client, profile, templates, n=n_target,
                                   use_fewshot=fewshot_on,
                                   extra_context=_phys_ctx)
            init_dsls.extend(batch)
            n_parsed_now = sum(
                1 for d in init_dsls
                if llm_ops.parse_dsl_safely(d, var_names) is not None
            )
            if n_parsed_now >= min_kept:
                break
    llm_stats["ginit"]["n_candidates"] = len(init_dsls)
    seen_dsl: set = set()
    for dsl in init_dsls:
        if dsl in seen_dsl:
            continue
        seen_dsl.add(dsl)
        g = llm_ops.parse_dsl_safely(dsl, var_names)
        if g is not None and g.count_edges() <= cfg.max_edges:
            if _gpo and g.case_names():
                # true-ASM: K# → C# 改写 (保留结构, 强制全局常数), 非拒绝
                _force_global_params(g)
                llm_stats["ginit"]["n_k_rewritten"] = (
                    llm_stats["ginit"].get("n_k_rewritten", 0) + 1)
            pop.append(g)
            llm_stats["ginit"]["n_parsed"] += 1
    while len(pop) < cfg.population:
        pop.append(gp_ops.random_graph(rng, list(var_names),
                                       cfg.max_edges, cfg.max_params, cfg.max_depth))
    pop = pop[:cfg.population]
    # P1d-B2: 壁面特征种子注入 — G_INIT 候选不足时程序化补种 (P1c 教训:
    # T1 minimax 冷启动就没引入 yw, 事后 PIN 拦不住近壁通道缺失)
    # v3 消融: cfg.wall_seeds=False 时关闭 (裸配置臂剥离该共享机制)
    if getattr(cfg, "wall_seeds", True):
        pop = _inject_wall_seeds(pop, cases, var_names, cfg)

    # 3) 主循环
    # V4 (2026-10-12) 效率重构 — 三个来源的浪费:
    #   (a) 旧版每代对全种群 (含未变动的存活者) 重跑 paramfit → budget 被吃光,
    #       GP 实际代数远低于 generations (lgf_mg/m 2834s ≫ 600s 的主因之一);
    #   (b) 每代重新随机子采样 → 跨代 loss 不可比;
    #   (c) gen0 的 30 个 G_INIT 候选与随机图同权做全参数拟合.
    # 改法: 整个 GP 阶段用一份固定子采样 (loss 跨代可比); gen0 全种群
    # 1-start 初筛 (screen_budget_s); 之后只拟合新个体 (loss=inf)。
    sub_cases = _subsample(cases, cfg.sub_sample, rng)   # 固定评估集
    best_loss = float("inf"); best_g: Optional[Graph] = None
    stagnation = 0; t0 = time.time()
    mguide_pending: List[Graph] = []       # V3-a: 上一轮 M_GUIDE 追加体, 待统计存活

    def _fit_one(g: Graph, *, n_starts: int, maxiter: int, budget_s: float):
        """带异常兜底的单图拟合; 返回是否成功."""
        try:
            fit = fit_joint(g, sub_cases, n_starts=n_starts,
                            maxiter=maxiter, init_range=cfg.param_init_range,
                            fit_tol=cfg.fit_tol,
                            time_budget_s=budget_s,
                            seed=int(rng.integers(0, 1_000_000)))
            g.loss = fit.loss; g.theta = fit.theta; g.metrics = fit.per_case
            return True
        except Exception:
            g.loss = float("inf")
            return False

    # gen0 初筛: 全种群 1-start 快速拟合 (FORBIDDEN 过滤同旧版)
    # V4-b: 初筛成功的个体打 _screened 标记 — 1-start 粗 loss 与进化期
    # gp_starts 全量拟合的新个体不公平竞争 (粗 loss 偏高 → 精英被低估),
    # 下一代对 _screened 个体补一次全量拟合后清标记 (每个体只补一次).
    for g in pop:
        if FORBIDDEN and llm_ops._has_op(g.root, FORBIDDEN):
            g.loss = float("inf")
            llm_stats["forbidden_filtered"] += 1
            continue
        ok = _fit_one(g, n_starts=1, maxiter=min(cfg.gp_maxiter, 30),
                      budget_s=float(getattr(cfg, "screen_budget_s", 3.0)))
        g._screened = ok
        if g.loss < best_loss:
            best_loss = g.loss; best_g = g.clone()
            stagnation = 0

    # V4-b (2026-10-12): GP 子代 loss/theta 重置.
    # 根因: Graph.clone() 拷贝父代 loss/theta (graph.py:189-195), 而
    # gp_ops.crossover/mutate 全部通过 clone 构造子代 -> 子代继承父代的
    # 缓存 loss. V4 存活者缓存 (下方 `if g.loss < inf and g.theta...`)
    # 误把刚生成的子代当作存活者跳过拟合 -> GP 进化完全失效 (每代 0.7s,
    # 40 代仅 0.408 -> 0.297, top_final 坍缩到 1-2 个结构家族).
    # 修复: 子代入种群前强制重置, 下一轮 eval 循环必然重拟合.
    def _reset_offspring(c):
        c.loss = float("inf")
        c.theta = None
        c._screened = False

    for gen in range(cfg.generations):
        # V4-b: 预算预留 25% 给最终精修, 否则 GP 吃掉全部 budget 后
        # final refit 每人只分到 1-2s, 8 starts 起步优化无从展开.
        if time.time() - t0 > 0.75 * cfg.time_budget_s:
            break
        # 评估种群: 只拟合新个体 (loss=inf); 存活者沿用缓存 loss/theta
        # V3-a: 上一代 M_GUIDE 变异体的存活结算 (仍在种群且 loss 有限)
        if mguide_pending:
            llm_stats["mguide"]["n_alive"] += sum(
                1 for g in mguide_pending if g in pop and g.loss < float("inf"))
            mguide_pending = []
        for g in pop:
            if (g.loss < float("inf") and g.theta is not None
                    and not getattr(g, "_screened", False)):
                continue                          # V4: 存活者缓存
            if FORBIDDEN and llm_ops._has_op(g.root, FORBIDDEN):
                g.loss = float("inf")
                llm_stats["forbidden_filtered"] += 1
                continue
            _fit_one(g, n_starts=cfg.gp_starts, maxiter=cfg.gp_maxiter,
                     budget_s=float(getattr(cfg, "gp_fit_budget_s", 10.0)))
            g._screened = False                  # V4-b: 补拟合完成, 清标记
            if g.loss < best_loss:
                best_loss = g.loss; best_g = g.clone()
                stagnation = 0
        # 选择
        survivors = gp_ops.select(pop, keep_ratio=0.5)
        offspring = []
        while len(survivors) + len(offspring) < cfg.population:
            if rng.random() < cfg.crossover_rate and len(survivors) >= 2:
                a, b = survivors[rng.integers(0, len(survivors))], survivors[rng.integers(0, len(survivors))]
                child = gp_ops.crossover(a, b, rng, cfg.max_edges, cfg.max_params, cfg.max_depth)
                if child is not None:
                    _reset_offspring(child)
                    offspring.append(child)
            if len(survivors) + len(offspring) >= cfg.population:
                break
            parent = survivors[rng.integers(0, len(survivors))]
            child = gp_ops.mutate(parent, rng, list(var_names),
                                   max_edges=cfg.max_edges,
                                   max_params=cfg.max_params,
                                   max_depth=cfg.max_depth)
            if child is not None:
                _reset_offspring(child)
                offspring.append(child)
        pop = survivors + offspring
        stagnation += 1

        # 外层: 每 K 代 + 停滞时
        # P1d-C2: llm_ginit_only 模式下 LLM 只出 G_INIT 初始种群, 进化全程离线
        if (cfg.use_llm and not getattr(cfg, "llm_ginit_only", False)
                and cfg.use_attribution and best_g is not None and
                (gen % cfg.llm_every == cfg.llm_every - 1 or stagnation >= cfg.stagnation_window)):
            try:
                rep = attribution.analyze(best_g, best_g.theta, sub_cases, var_names)
                # P1d-B3: 目标级禁用算子注入报告 → M_GUIDE prompt 硬规则
                rep["forbidden_ops"] = list(FORBIDDEN)
                llm_stats["mguide"]["n_calls"] += 1
                mutations = llm_ops.M_GUIDE(llm_client, [(best_g, best_g.loss)], rep,
                                            extra_context=_phys_ctx)
                if not mutations:
                    # V3-a: 空建议时记录最近一次后端失败 (API 挂 vs 返回体无 mutations 键)
                    err = dict(llm_ops.LAST_ERROR)
                    if err.get("error"):
                        llm_stats["mguide"]["last_error"] = (
                            f"gen{gen}: {err.get('backend')} {err.get('url')}: "
                            f"{err.get('error')}")
                else:
                    llm_stats["mguide"]["n_nonempty"] += 1
                if mutations:
                    # 应用到 top 5 个体
                    # P1d-B1 精英保护: 变异体重置 loss 后**追加**进种群参与竞争,
                    # 不再顶替精英 (旧实现 new_top + pop[5:] 会把原图直接丢掉,
                    # 一次坏建议即毁精英层; 下代全种群重评 + select 自然淘汰劣者)
                    new_top = []
                    for g in sorted(pop, key=lambda x: x.loss)[:5]:
                        ng = llm_ops.apply_mutations(g, mutations, rng,
                                                     forbidden_ops=list(FORBIDDEN))
                        if (ng is not None and ng.count_edges() <= cfg.max_edges
                                and ng.count_params() <= cfg.max_params):
                            if _gpo and ng.case_names():
                                # true-ASM: 变异注入的 K# → C# 改写 (ng 为克隆, 安全)
                                _force_global_params(ng)
                                llm_stats["mguide"]["n_k_rewritten"] = (
                                    llm_stats["mguide"].get("n_k_rewritten", 0) + 1)
                            ng.loss = float("inf")   # 不继承父代 loss, 下代重评
                            new_top.append(ng)
                    if new_top:
                        llm_stats["mguide"]["n_applied"] += len(new_top)
                        pop = pop + new_top
                        mguide_pending = list(new_top)
            except Exception as e:
                # V3-a: 不再静默吞掉 — traceback 进 llm_stats + 单行提示到 stdout
                import traceback as _tb
                tb = _tb.format_exc()
                llm_stats["mguide"]["last_exception"] = f"gen{gen}: {tb[-2000:]}"
                print(f"[lgf][V3a] M_GUIDE 链路异常 gen{gen}: {type(e).__name__}: {e}",
                      flush=True)

        if stagnation >= cfg.stagnation_window:
            pop = gp_ops.extinction(pop, keep=cfg.extinction_keep, rng=rng,
                                     var_names=list(var_names),
                                     max_edges=cfg.max_edges,
                                     max_params=cfg.max_params,
                                     max_depth=cfg.max_depth,
                                     templates=list(TEMPLATE_REGISTRY.keys()))
            stagnation = 0

        genlog.append(GenLog(gen=gen, best_loss=best_loss,
                              best_graph=(best_g.to_dsl(with_poly_defs=False)
                                          if best_g else ""),
                              n_pop=len(pop),
                              seconds=time.time() - t0).__dict__)
        if gen % cfg.log_every == 0:
            _append_jsonl(log_path, genlog[-1])

    # 4) 最终精修 top-K
    top = sorted(pop, key=lambda x: x.loss)[:8]
    refined: List[Tuple[Graph, float, np.ndarray, Dict]] = []
    for g in top:
        if FORBIDDEN and llm_ops._has_op(g.root, FORBIDDEN):
            continue          # V3-b: 精修同样排除禁用算子结构
        if _gpo and g.case_names():
            _force_global_params(g)   # true-ASM: 兜底改写 (理论上不会触发)
        try:
            fit = fit_joint(g, cases, n_starts=cfg.final_starts,
                             maxiter=cfg.final_maxiter, init_range=cfg.param_init_range,
                             fit_tol=cfg.fit_tol,
                             time_budget_s=max(1.0, (cfg.time_budget_s - (time.time() - t0)) / max(1, len(top))),
                             seed=int(rng.integers(0, 1_000_000)))
            g.loss = fit.loss; g.theta = fit.theta; g.metrics = fit.per_case
            refined.append((g.clone(), fit.loss, fit.theta, fit.per_case))
        except Exception:
            continue
    refined.sort(key=lambda r: r[1])

    # 5) 等价判定 (符号 + 数值) — 在 top 上做 (refined 按损失升序, 取最优前 8)
    if not refined:
        # 兜底: 精修全部失败时直接使用种群 top, 避免空列表 IndexError
        for g in top:
            refined.append((g.clone(), g.loss, g.theta, g.metrics))
        refined.sort(key=lambda r: r[1])
    final_top = refined[:8]
    final = {
        "best": {"dsl": final_top[0][0].to_dsl(with_poly_defs=False),
                 "loss": final_top[0][1],
                 "theta": final_top[0][2].tolist(),
                 "param_names": [p.name for p in final_top[0][0].params()],
                 "metrics": final_top[0][3],
                 "complexity": final_top[0][0].complexity()},
        "top_final": [{
                "dsl": f[0].to_dsl(with_poly_defs=False),
                "loss": f[1],
                "theta": f[2].tolist(),
                "param_names": [p.name for p in f[0].params()],
                "metrics": f[3],
                "complexity": f[0].complexity()} for f in final_top],
        "log": genlog,
        "profile": profile,
        "llm_stats": llm_stats,
        "priors": (prior_constraints.to_dict() if prior_constraints is not None else None),
        "elapsed_s": time.time() - t0,
        "cfg": _mask_cfg(cfg),
        "name": name,
    }
    with open(os.path.join(log_dir, f"{name}_final.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in final.items() if k != "log"}, f,
                   ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(log_dir, f"{name}_log.jsonl"), "w", encoding="utf-8") as f:
        for entry in genlog:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return final


def _mask_cfg(cfg: LGFConfig) -> Dict:
    """cfg 序列化 + 敏感字段脱敏 (api key 不落盘, 2026-10-08 v3)."""
    d = cfg.to_dict()
    if d.get("llm_api_key"):
        d["llm_api_key"] = "***masked***"
    return d


def _quick_profile(cases: List[dict], var_names: Sequence[str]) -> Dict:
    from .priors import data_profile
    return data_profile(cases, list(var_names))


def _inject_wall_seeds(pop: List[Graph], cases: List[dict], var_names: Sequence[str],
                       cfg: LGFConfig, min_count: int = 3) -> List[Graph]:
    """P1d-B2: 壁面特征 (yw) 种子注入.

    若种群中含壁面特征的个体不足 min_count, 程序化补 3 个结构化种子:
      C1·f^2·exp(C2/yw),  C1·exp(C2/yw),  C1·f^2/(1+exp(C2/yw))
    f 取与目标 y 相关性最高的不变量. 解析失败/超复杂度的种子直接丢弃.

    注意: 传入的 pop 已裁剪到 cfg.population, 这里允许临时超出 (±3),
    下一轮 select 会自然回到规模 — 换掉的是随机填充个体, 不是精英.
    """
    from .graph import walk
    wall = "yw" if "yw" in var_names else ("ypd" if "ypd" in var_names else None)
    if wall is None:
        return pop

    def _has_wall(g: Graph) -> bool:
        return any(n.op == "var" and 0 <= n.attrs.get("index", -1) < len(g.var_names)
                   and g.var_names[n.attrs["index"]] == wall
                   for _, n in walk(g.root))

    if sum(1 for g in pop if _has_wall(g)) >= min_count:
        return pop

    # 与目标 y 相关性最高的特征 f
    best_f, best_c = None, -1.0
    try:
        Y = np.concatenate([np.asarray(c["y"], dtype=float) for c in cases])
        X = np.concatenate([np.asarray(c["X"], dtype=float) for c in cases], axis=0)
        m = np.isfinite(Y) & np.isfinite(X).all(axis=1)
        Y, X = Y[m], X[m]
        for j, nm in enumerate(var_names):
            if nm == wall:
                continue
            x = X[:, j]
            if float(np.std(x)) < 1e-12:
                continue
            c = abs(float(np.corrcoef(x, Y)[0, 1]))
            if np.isfinite(c) and c > best_c:
                best_c, best_f = c, nm
    except Exception:
        best_f = None
    f = best_f or next((nm for nm in var_names if nm != wall), None)
    if f is None:
        return pop

    seeds = [
        f"C1 * {f}^2 * exp(C2/{wall})",
        f"C1 * exp(C2/{wall})",
        f"C1 * {f}^2 / (1 + exp(C2/{wall}))",
    ]
    added = 0
    for s in seeds:
        g = llm_ops.parse_dsl_safely(s, list(var_names))
        if g is None:
            continue
        if g.count_edges() > cfg.max_edges or g.count_params() > cfg.max_params:
            continue
        pop.append(g)
        added += 1
    if added:
        print(f"[lgf][B2] wall-seed: 注入 {added} 个含 {wall} 的种子 "
              f"(f={f}, corr={best_c:.2f}, 种群含{wall}个体数 "
              f"{sum(1 for g in pop if _has_wall(g))}/{len(pop)})", flush=True)
    return pop


def _subsample(cases: List[dict], n: int, rng) -> List[dict]:
    out = []
    for c in cases:
        if c["X"].shape[0] <= n:
            d = dict(X=c["X"], y=c["y"], name=c.get("name", "case"))
            if "w" in c:
                d["w"] = c["w"]
            out.append(d)
            continue
        idx = rng.choice(c["X"].shape[0], n, replace=False)
        d = dict(X=c["X"][idx], y=c["y"][idx], name=c.get("name", "case"))
        if "w" in c:
            d["w"] = c["w"][idx]
        out.append(d)
    return out


def _append_jsonl(path: str, entry: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ============================================================================
# 顶层 main 入口 (便于脚本与 main.py 调用)
# ============================================================================
def main(cases: List[dict], var_names: Sequence[str], cfg: Optional[LGFConfig] = None,
         log_dir: str = "outputs/run", name: str = "run", llm_client=None,
         benchmark: Optional[Benchmark] = None) -> Dict:
    if cfg is None:
        cfg = LGFConfig()
    if llm_client is None:
        if getattr(cfg, "llm_debug", False):
            llm_ops.set_debug(True)
        llm_client = llm_ops.make_client(cfg.llm_backend, url=cfg.llm_url,
                                          model=cfg.llm_model,
                                          api_key=cfg.llm_api_key,
                                          temperature=cfg.llm_temperature,
                                          timeout=cfg.llm_timeout)
    return run_lgf(cases, var_names, cfg, log_dir, llm_client, name=name,
                   profile=None if benchmark is None else None)