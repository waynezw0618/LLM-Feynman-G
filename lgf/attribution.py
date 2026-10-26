"""lgf.attribution — 误差归因, 把残差归到具体子图节点, 用于驱动 LLM 定向变异.

对每个算子节点 v:
  - 提取子表达式 z_v = subtree_v(theta; X)
  - 与残差 r = y - y_pred 做 Pearson |corr(z_v, r)| 与 Spearman |corr|
  - 偏依赖: 按 z_v 分 10 bin, 取每 bin 的平均残差 → 形状标签

物理角色标注 (L1 prompt 注入需要):
  - yw: 真壁面距离 (1-|y/h|)·Re_tau (wall units), 不是雷诺应力不变量
        → 必含项, 禁止被全部替换 (近壁阻尼 φ(yw) 必要; yw=0 壁面, yw=180 中心)
  - ypd: [已弃用] 旧名 |y+|=|y/h|·Re_tau 实为离中心距离, 仅向后兼容保留
  - eta2 / s3: 共享 Reynolds 不变量 (v/e 通用)
  - zeta2v / so2v / s2o2v: 涡量版不变 (旋转线性部分)
  - zeta2e / so2e / s2o2e: 扩展内禀不变 (含 2Ω 修正, 旋转耦合)
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .graph import Graph, walk


# 特征物理角色字典 (供 LLM prompt 注入)
# - role: "wall_distance" | "invariant" | "rotation_coupling"
# - pinned: True 表示该特征不应在 replace_subtree 中被全部替换
FEATURE_ROLES: Dict[str, Dict[str, str]] = {
    "yw":     {"role": "wall_distance",   "pinned": "true",  "desc": "真壁面距离 (1-|y/h|)·Re_tau wall units (yw=0 壁面, yw=180 中心), 不是 Reynolds 不变量, 近壁阻尼 φ(yw) 必要"},
    "ypd":    {"role": "wall_distance",   "pinned": "true",  "desc": "[已弃用旧名] |y⁺| 离中心距离, 语义与 yw 互补, 近壁阻尼请优先用 yw"},
    "eta2":   {"role": "invariant",       "pinned": "false", "desc": "应变强度 tr(S²), v/e 共享"},
    "s3":     {"role": "invariant",       "pinned": "false", "desc": "tr(S³) 三阶不变量, v/e 共享"},
    "zeta2v": {"role": "rotation",        "pinned": "false", "desc": "−tr(W²) 涡量版旋转强度"},
    "so2v":   {"role": "rotation",        "pinned": "false", "desc": "tr(S·W²) 涡量版旋转耦合"},
    "s2o2v":  {"role": "rotation",        "pinned": "false", "desc": "tr(S²:W²) 涡量版旋转-应变耦合"},
    "zeta2e": {"role": "rotation_coupling","pinned": "false", "desc": "−tr(W̄²) 扩展内禀 (W̄ = W + 2Ω), 含帧旋转"},
    "so2e":   {"role": "rotation_coupling","pinned": "false", "desc": "tr(S·W̄²) 扩展内禀"},
    "s2o2e":  {"role": "rotation_coupling","pinned": "false", "desc": "tr(S²:W̄²) 扩展内禀"},
    "zeta2g": {"role": "rotation_coupling","pinned": "false", "desc": "−tr(W*²) Gatski-Speziale 修正旋转 (W* = W + 1·Ω), 帧旋转以 1 倍进入"},
    "so2g":   {"role": "rotation_coupling","pinned": "false", "desc": "tr(S·W*²) Gatski-Speziale 修正旋转"},
    "s2o2g":  {"role": "rotation_coupling","pinned": "false", "desc": "tr(S²:W*²) Gatski-Speziale 修正旋转"},
    "ret":    {"role": "turbulence_state", "pinned": "false", "desc": "局部湍流雷诺数 Re_t=k²/(ν ε) (ν=1 粘性单位). 低-Re 近壁阻尼变量 (Myon-Kasagi f_μ 型), y⁺∈[1,30] 跨 4 个量级, 平台 188-317 随 Ro_τ 饱和. 非 Reynolds 不变量但是允许的局部标量扩展 (与 yw 同级合法), 禁止以'非不变量'为由从公式中删除"},
    "roloc":  {"role": "rotation",        "pinned": "false", "desc": "局部旋转数 2Ωk/ε (带符号, 本流动族 5 工况同号), 中心值 5-127 随 Ro_τ. 与 zeta2e 的区别: 显式旋转强度, 不与 dW/dy 剪切纠缠"},
}


def analyze(graph: Graph, theta: np.ndarray, cases: List[dict],
           var_names: Sequence[str]) -> Dict:
    """对每个非叶节点计算归因. 用 sympy/lambdify 提取中间表达式."""
    expr = graph.to_sympy()
    var_syms = [sp.Symbol(n, real=True) for n in var_names]
    g_syms = [sp.Symbol(p.name, real=True) for p in graph.params()
              if p.kind == "global"]
    c_syms = [sp.Symbol(p.name, real=True) for p in graph.params()
              if p.kind == "case"]
    P_g = len(g_syms); P_c = len(c_syms)

    # 子树表达式表 (path -> sympy expr)
    sub_exprs: Dict[Tuple[int, ...], sp.Expr] = {}
    for path, n in walk(graph.root):
        if n.op in ("var", "const"):
            sub_exprs[path] = None  # leaf 跳过归因
            continue
        # 构造子树 expr (递归收集 var/const/params)
        def E(node):
            if node.op == "var":
                return var_syms[node.attrs["index"]]
            if node.op == "const":
                return sp.Float(node.attrs["const"])
            args = []
            for ch, pp in node.children:
                ce = E(ch)
                if pp is not None:
                    ce = ce * sp.Symbol(pp.name, real=True)
                args.append(ce)
            op = node.op
            if op == "add": return sp.Add(*args)
            if op == "mul": return sp.Mul(*args)
            if op == "sub": return args[0] - args[1]
            if op == "div": return args[0] / args[1]
            if op == "powi": return args[0] ** int(node.attrs["power"])
            if op == "inv": return sp.Integer(1) / args[0]
            if op == "neg": return -args[0]
            if op == "sqrt": return sp.sqrt(args[0])
            if op == "exp": return sp.exp(args[0])
            if op == "log": return sp.log(args[0])
            if op == "sin": return sp.sin(args[0])
            if op == "cos": return sp.cos(args[0])
            if op == "tanh": return sp.tanh(args[0])
            raise ValueError(op)
        sub_exprs[path] = E(n)

    # lambdify 每个子表达式
    sym_order = [*var_syms, *g_syms, *c_syms]
    sub_fns = {p: sp.lambdify(sym_order, e, "numpy")
               for p, e in sub_exprs.items() if e is not None}

    # 评估节点值
    nodes_data: Dict[Tuple[int, ...], Dict[str, float]] = {}
    resids = []; yps = []
    yps_per_case: List[np.ndarray] = []   # L2: 用于 ypd bin 误差分解
    for ci, case in enumerate(cases):
        X = case["X"]; y = case["y"]
        tl = np.concatenate([theta[:P_g], theta[P_g + ci*P_c : P_g + (ci+1)*P_c]])
        args = [X[:, i] for i in range(X.shape[1])] + list(tl)
        for path, fn in sub_fns.items():
            try:
                z = np.asarray(fn(*args), dtype=float)
            except Exception:
                z = np.full(X.shape[0], np.nan)
            # 常数子树返回 0 维标量 → 广播为 (N,)
            if z.ndim == 0:
                z = np.full(X.shape[0], float(z))
            elif z.shape != (X.shape[0],):
                z = np.full(X.shape[0], np.nan)
            if path not in nodes_data:
                nodes_data[path] = {"z": [], "r": []}
            nodes_data[path]["z"].append(z)
        # 总预测
        yp = np.asarray(sp.lambdify(sym_order, expr, "numpy")(*args), dtype=float)
        yps.append(yp)
        resids.append(y - yp)
        yps_per_case.append(yp)
    yps_all = np.concatenate(yps) if yps else np.array([])
    rs_all = np.concatenate(resids) if resids else np.array([])
    rmse = float(np.sqrt(np.mean(rs_all ** 2)))

    # 计算归因
    attributions: Dict[Tuple[int, ...], Dict] = {}
    for path, d in nodes_data.items():
        z = np.concatenate(d["z"])
        # |corr| 用 Pearson; Spearman via rank (剔除非有限值)
        m = np.isfinite(z) & np.isfinite(rs_all)
        zf, rf = z[m], rs_all[m]
        try:
            from scipy.stats import pearsonr, spearmanr
            pr = pearsonr(zf, rf)[0] if (np.std(zf) > 1e-9 and len(zf) > 2) else 0.0
            sr = spearmanr(zf, rf).correlation if hasattr(spearmanr(zf, rf), "correlation") else spearmanr(zf, rf)[0]
        except Exception:
            pr, sr = 0.0, 0.0
        pr = float(pr) if np.isfinite(pr) else 0.0
        sr = float(sr) if np.isfinite(sr) else 0.0
        # 偏依赖 / 形状
        shape = _shape_label(zf, rf)
        attributions[path] = dict(
            pearson=float(pr), spearman=float(sr),
            sensitivity=float(abs(pr)),
            shape=shape,
            std=float(np.std(zf)) if len(zf) else 0.0,
        )

    # 排序: 敏感度高且拟合形态不匹配 (形状 != 'flat' & !monotone match with y) 优先
    targets = sorted(attributions.keys(),
                    key=lambda p: -attributions[p]["sensitivity"])[:5]

    # L1 物理角色 + L2 壁面距离 bin 误差分解 (优先 yw, 兼容旧 ypd)
    feature_roles = {n: FEATURE_ROLES.get(n, {"role": "unknown", "pinned": "false",
                                              "desc": ""})
                     for n in var_names}
    wall_feat = "yw" if "yw" in var_names else ("ypd" if "ypd" in var_names else None)
    ypd_bin_errors = _ypd_bin_error_share(cases, var_names,
                                          ypds=[X[:, list(var_names).index(wall_feat)]
                                                if wall_feat is not None else None
                                                for X in [c.get("X") for c in cases]],
                                          y_preds_per_case=yps_per_case,
                                          wall_feat=wall_feat)

    # P1d-A1/A2/A3: 逐 case 误差画像 + 目标值域语义 (喂给 M_GUIDE 的物理先验输入)
    per_case = _per_case_summary(cases, var_names, yps_per_case)
    target_sem = _target_semantics(cases, var_names)

    return dict(
        rmse=rmse,
        per_node=attributions,
        suggested_targets=[list(t) for t in targets],
        text_summary=_text_summary(attributions, targets,
                                   feature_roles, ypd_bin_errors,
                                   per_case, target_sem),
        feature_roles=feature_roles,
        ypd_bin_errors=ypd_bin_errors,
        per_case_summary=per_case,
        target_semantics=target_sem,
    )


def _per_case_summary(cases: List[dict], var_names: Sequence[str],
                      y_preds_per_case: Sequence[np.ndarray]) -> List[Dict]:
    """P1d-A1: 逐 case (逐 Ro 工况) 误差画像.

    让 M_GUIDE 看到 "哪个 Ro 工况在哪种区域 (近壁/中心) 以何种方式失败",
    而不是只看 5 工况聚合后的单一标量. 区域按壁面特征 wall units 分:
    near < 30, core > 120 (yw=0 壁面, 180 中心); 无壁面特征时 share 为 None.
    """
    out: List[Dict] = []
    wall_feat = "yw" if "yw" in var_names else ("ypd" if "ypd" in var_names else None)
    wf_idx = list(var_names).index(wall_feat) if wall_feat is not None else None
    for ci, case in enumerate(cases):
        if ci >= len(y_preds_per_case) or case.get("y") is None:
            continue
        y = np.asarray(case["y"], dtype=float)
        yp = np.asarray(y_preds_per_case[ci], dtype=float)
        m = np.isfinite(y) & np.isfinite(yp)
        y, yp = y[m], yp[m]
        if y.size == 0:
            continue
        nmse = float(np.mean((yp - y) ** 2) / (np.var(y) + 1e-12))
        near_share = core_share = None
        if wf_idx is not None and case.get("X") is not None:
            w = np.abs(np.asarray(np.asarray(case["X"])[m][:, wf_idx], dtype=float))
            tot = float(np.sum((yp - y) ** 2)) + 1e-12
            near_share = float(np.sum((yp - y)[w < 30] ** 2) / tot)
            core_share = float(np.sum((yp - y)[w > 120] ** 2) / tot)
        # 符号翻转 (大幅值点上): sub/div 结构值域错的直接症状
        amp = float(np.max(np.abs(y)))
        sig = np.abs(y) > 0.2 * amp
        flip_frac = (float(np.mean(np.sign(yp[sig]) != np.sign(y[sig])))
                     if int(sig.sum()) > 3 else 0.0)
        labels = []
        if nmse > 1.0:
            labels.append("EXPLODED")
        elif nmse > 0.3:
            labels.append("POOR")
        if flip_frac > 0.05:
            labels.append("SIGN-FLIP")
        if near_share is not None and near_share > 0.5:
            labels.append("NEAR-WALL-DOMINATED")
        elif core_share is not None and core_share > 0.5:
            labels.append("CORE-DOMINATED")
        out.append(dict(name=str(case.get("name", f"case{ci}")), nmse=nmse,
                        near_share=near_share, core_share=core_share,
                        flip_frac=flip_frac, label="|".join(labels) or "OK"))
    return out


def _target_semantics(cases: List[dict], var_names: Sequence[str]) -> Dict:
    """P1d-A2: 目标 y 与各特征的值域语义 — M_GUIDE 硬规则的自动输入.

    - y_min/y_max/frac_neg: 目标值域与符号 (非负目标上 sub/div 有值域风险)
    - vanishing: |x|≈0 点占比 > 1% 的特征 = 危险分母清单
    """
    Ys, Xs = [], []
    for c in cases:
        if c.get("y") is None:
            continue
        Ys.append(np.asarray(c["y"], dtype=float))
        Xs.append(np.asarray(c["X"], dtype=float))
    if not Ys:
        return {}
    y = np.concatenate(Ys)
    X = np.concatenate(Xs, axis=0)
    yf = y[np.isfinite(y)]
    sem: Dict = {}
    if yf.size:
        sem["y_min"] = float(np.min(yf))
        sem["y_max"] = float(np.max(yf))
        sem["frac_neg"] = float(np.mean(yf < 0))
    van: Dict[str, float] = {}
    for j, name in enumerate(var_names):
        x = X[np.isfinite(X[:, j]), j]
        if x.size == 0:
            continue
        scale = max(float(np.ptp(x)), 1e-12)
        frac = float(np.mean(np.abs(x) < 1e-3 * scale))
        if frac > 0.01:
            van[str(name)] = frac
    if van:
        sem["vanishing"] = van
    return sem


def _ypd_bin_error_share(cases: List[dict], var_names: Sequence[str],
                         ypds: Optional[Sequence[np.ndarray]] = None,
                         y_preds_per_case: Optional[Sequence[np.ndarray]] = None,
                         wall_feat: Optional[str] = None
                         ) -> List[Dict]:
    """按壁面距离分 4 个 bin 汇报各 bin 误差占总误差百分比.

    wall_feat: "yw" (真壁面距离, yw=0 壁面) 或 "ypd" (旧离中心距离, 仅兼容).
    bin: [0,15), [15,40), [40,80), [80,inf)   (wall units)
    返回 [{'bin':'[0,15)','share':0.6,'mean_resid':0.12,'n':200}, ...]
    若数据中无壁面距离列, 返回 [].
    """
    if wall_feat is None:
        wall_feat = "yw" if "yw" in var_names else ("ypd" if "ypd" in var_names else None)
    if wall_feat is None or not y_preds_per_case:
        return []
    ypd_idx = list(var_names).index(wall_feat)
    bins = [(0, 15), (15, 40), (40, 80), (80, np.inf)]
    out: List[Dict] = []
    for ci, case in enumerate(cases):
        X = case.get("X"); y = case.get("y")
        if X is None or y is None or ci >= len(y_preds_per_case):
            continue
        if ypds is not None and ypds[ci] is not None:
            yp = np.abs(np.asarray(ypds[ci], dtype=float))
        else:
            yp = np.abs(np.asarray(X[:, ypd_idx], dtype=float))
        if yp.size == 0:
            continue
        r = np.asarray(y, dtype=float) - np.asarray(y_preds_per_case[ci], dtype=float)
        total = float(np.sum(r ** 2))
        if total <= 0:
            continue
        for lo, hi in bins:
            mask = (yp >= lo) & (yp < hi)
            if not np.any(mask):
                continue
            out.append({
                "case_idx": ci,
                "bin": f"[{lo},{'inf' if hi == np.inf else hi})",
                "share": float(np.sum(r[mask] ** 2) / total),
                "mean_abs_resid": float(np.mean(np.abs(r[mask]))),
                "n": int(mask.sum()),
            })
    # 按 bin 聚合 (case 之间平均)
    if not out:
        return []
    bins_unique = sorted({o["bin"] for o in out}, key=lambda s: float(s.split(",")[0].lstrip("[")))
    agg = []
    for b in bins_unique:
        sel = [o for o in out if o["bin"] == b]
        agg.append({
            "bin": b,
            "share": float(np.mean([o["share"] for o in sel])),
            "mean_abs_resid": float(np.mean([o["mean_abs_resid"] for o in sel])),
            "n": int(sum(o["n"] for o in sel)),
        })
    return agg


def _shape_label(z: np.ndarray, r: np.ndarray) -> str:
    if np.std(z) < 1e-9:
        return "constant"
    # 分 bin 求残差均值
    nb = min(10, max(2, len(z) // 20))
    order = np.argsort(z)
    z_sorted = z[order]; r_sorted = r[order]
    bins = np.array_split(np.arange(len(z)), nb)
    means = np.array([np.mean(r_sorted[b]) for b in bins])
    rng = max(np.ptp(means), 1e-12)
    if rng < 0.01 * (np.std(r) + 1e-9):
        return "flat"
    slope_sign = np.sign(means[-1] - means[0])
    monotone_frac = (np.diff(means) * slope_sign > -1e-9).mean() if len(means) >= 2 else 1.0
    if monotone_frac > 0.7:
        return "increasing" if slope_sign > 0 else "decreasing"
    if np.argmax(np.abs(means)) in (0, len(means) - 1):
        return "saturating"
    return "nonmonotone"


def _text_summary(attrib: Dict, targets: List[Tuple[int, ...]],
                 feature_roles: Optional[Dict[str, Dict]] = None,
                 ypd_bin_errors: Optional[List[Dict]] = None,
                 per_case: Optional[List[Dict]] = None,
                 target_sem: Optional[Dict] = None) -> str:
    lines = ["[误差归因报告]"]
    for t in targets:
        a = attrib[t]
        path_s = "/".join(str(x) for x in t)
        lines.append(f"  节点 {path_s}: sensitivity={a['sensitivity']:.3f}, "
                     f"shape={a['shape']}, std={a['std']:.3g}")

    # L1: 物理角色注解 (LLM 看到这是不变量还是壁面距离)
    if feature_roles:
        lines.append("[特征物理角色]")
        for name, info in feature_roles.items():
            pin = "🔒PIN" if str(info.get("pinned", "false")).lower() == "true" else "  "
            lines.append(f"  {pin} {name}: role={info.get('role','?')}, "
                         f"desc={info.get('desc','')}")

    # L2: 壁面区误差分解 (LLM 看到近壁误差占大头 → 不该删 ypd)
    if ypd_bin_errors:
        lines.append("[近壁误差分布 |ypd| bins]")
        for e in ypd_bin_errors:
            lines.append(f"  bin {e['bin']}: error_share={e['share']:.1%}, "
                         f"mean_|resid|={e['mean_abs_resid']:.3g}, n={e['n']}")
        # 标注近壁占比
        near = sum(e["share"] for e in ypd_bin_errors if e["bin"].startswith("[0,"))
        if near > 0.4:
            lines.append(f"  ⚠ 近壁 |ypd|<15 误差占 {near:.1%}, "
                         f"任何删除 ypd 节点的变异都应避免")

    # P1d-A2: 目标值域语义 (硬规则输入)
    if target_sem:
        lines.append("[目标值域语义]")
        y_min, y_max = target_sem.get("y_min"), target_sem.get("y_max")
        frac_neg = target_sem.get("frac_neg")
        if y_min is not None and frac_neg is not None:
            if frac_neg < 1e-3:
                sign_note = "恒非负"
            elif frac_neg > 0.99:
                sign_note = "恒非正"
            else:
                sign_note = f"{frac_neg:.0%} 为负 (混合符号)"
            lines.append(f"  y ∈ [{y_min:.3g}, {y_max:.3g}] ({sign_note}); "
                         f"变异后的公式必须保持该值域")
        for name, frac in (target_sem.get("vanishing") or {}).items():
            lines.append(f"  ⚠ 特征 {name} 有 {frac:.0%} 的点 |x|≈0 — "
                         f"放在分母 (div/inv) 会在这些点上爆炸")

    # P1d-A1/A3: 逐工况误差表 + 最差工况
    if per_case:
        lines.append("[逐工况误差]")
        for pc in per_case:
            reg = ""
            if pc["near_share"] is not None:
                reg = (f", 近壁share={pc['near_share']:.0%}, "
                       f"中心share={pc['core_share']:.0%}")
            if pc["flip_frac"] > 0.05:
                reg += f", 符号翻转{pc['flip_frac']:.0%}"
            lines.append(f"  {pc['name']}: NMSE={pc['nmse']:.3g}{reg} "
                         f"[{pc['label']}]")
        worst = max(per_case, key=lambda p: p["nmse"])
        if worst["nmse"] > 0.3:
            lines.append(f"  ⚠ WORST CASE {worst['name']} NMSE={worst['nmse']:.3g} "
                         f"({worst['label']}) — 任何变异不得使最差工况进一步恶化")
    return "\n".join(lines)