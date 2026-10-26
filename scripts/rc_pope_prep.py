"""rc_pope_prep.py — Pope (1975) 通用涡黏性框架数据管线 (阶段1b).

框架 (Pope, JFM 72:331, 1975; 亦见 Pope《Turbulent Flows》第11章):
  b_ij = sum_m a_m T_ij^(m),  a_m = f(I_i)
  10 个对称无迹张量基 T^(1..10) 由 (S, W) 构造; 标量系数依赖 5 个不变量.

本脚本扩展: W ∈ {Wbar (涡量), Omega_tilde (扩展内禀, 黄/马 2008 式(19), 系数2),
  Wgs (Gatski-Speziale 1993, W* = Wbar + eps*Omega, 系数1)} 三种取法
  -> 30 组候选基 (T1 与 W 无关, 去重后 28 个唯一);
  -> 15 个不变量特征 (每种 W 各 5 个: eta2/zeta2/s3/so2/s2o2, 其中 eta2/s3 共享
     -> 11 个唯一特征列).
  GS 与黄/马的区别: 系统旋转以 1 倍 (而非 2 倍) 进入旋转率张量,
  即 W*_ij = W_ij + eps_ijk*Om_k, 参见 Gatski & Speziale (1993) JFM 254:59.

基底选择 (不限制在旧 ASM 的 5 基, 也不必全选):
  贪心前向选择 (最小化全局相对残差) + 条件数换基改善 (残差不变坏的前提下
  降 med cond). K <= 5 (逐点对称无迹空间维数上限, 保证 a_m 逐点唯一).

输出 dataset_pope.npz 兼容 rc_asm_run/rc_asm_eval:
  - sel_basis: 选中基底名 (如 ["T1","T2v","T3",...])
  - featB_keys: 8 个不变量特征名
  - c{i}_T_{name}: 选中基张量 6 分量; c{i}_W_{name}: 逐点权重 ||T_m||_F^2
    (供 rc_asm_run --fitness stress 使用, 即"重建误差进 fitness")
  - c{i}_B_{name}: 全部 19 个候选基 (备重新筛选)

用法: python scripts/rc_pope_prep.py [--out dataset_pope2.npz] [--kmax 5]
       [--cond_max 1e4] [--amax_cap 300]
       [--p90_slack 200] [--nw_drop 3]
v2 (2026-09-30, P0-B): min-norm lstsq + valid_fit 双掩码 (cond_max/amax_cap),
       选基换基目标 med cond → p90 cond (RC2 近壁病态不再被中位数掩盖),
       输出候选子集对比表 (T6v/T10v/T9v vs T5v, RC3 vw 支撑量纲).
v3 (2026-10-04): 换基二段式判据 (p90 放松方案):
       段1 (现行, 不变): rel_resid≤2× 且 p90 < 0.7×base → 换
       段2 (--p90_slack N 启用): rel_resid≤2× 且 p90 < N×base, 且
         **coverage 守门** (用 p90 所代理的直接量, 不用代理量本身):
           nw_fit% (近壁 |y/h|>0.9 通过双掩码的占比) ≥ base − nw_drop (默认 3pp)
           且 max|a|@fit ≤ amax_cap
       两段顺序执行 (先收敛现行判据, 再探索放松), visited 集防振荡;
       末尾输出"放松可行邻域表"(所有过 rel_resid+coverage 守门的单换基邻居),
       供 T5v/vw 等专项决策. 默认 --p90_slack None → 行为与 v2 完全一致.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_prep import load_case, sym6, RO_CASES, COMP6, OUT_DIR, FIG_DIR

# 11 个唯一不变量特征 (v=vort 涡量, e=ext 扩展内禀[系数2], g=GS[系数1];
# eta2/s3 三者共享)
FEAT = ["eta2", "s3", "zeta2v", "so2v", "s2o2v", "zeta2e", "so2e", "s2o2e",
        "zeta2g", "so2g", "s2o2g"]


def pope_tensors(Ss, Ws):
    """Pope 1975 十张量基 (S, W 均已乘 k/eps). 返回 [(name, (N,3,3))]."""
    I = np.eye(3)[None, :, :]
    tr = lambda M: np.einsum("nii->n", M)
    tl = lambda M: M - (tr(M) / 3.0)[:, None, None] * I
    sym = lambda M: 0.5 * (M + np.transpose(M, (0, 2, 1)))
    S2 = Ss @ Ss
    W2 = Ws @ Ws
    SW2 = Ss @ W2
    S2W2 = S2 @ W2
    raw = [
        ("T1", Ss.copy()),
        ("T2", Ss @ Ws - Ws @ Ss),
        ("T3", S2.copy()),
        ("T4", W2.copy()),
        ("T5", Ws @ S2 - S2 @ Ws),
        ("T6", W2 @ Ss + Ss @ W2),
        ("T7", Ws @ Ss @ W2 - W2 @ Ss @ Ws),
        ("T8", Ss @ Ws @ S2 - S2 @ Ws @ Ss),
        ("T9", W2 @ S2 + S2 @ W2),
        ("T10", Ws @ S2 @ W2 - W2 @ S2 @ Ws),
    ]
    out = []
    for i, (n, T) in enumerate(raw):
        T = sym(T)
        if n in ("T3", "T4", "T6", "T9"):  # 需去迹的四个
            T = tl(T)
        # T1: 不可压缩 S 本身无迹; T2/T5/T7/T8/T10 解析无迹, 数值残迹可忽略
        out.append((n, T))
    return out


def invariants(Ss, Ws):
    e = lambda M: np.einsum("nii->n", M)
    S2 = Ss @ Ss
    O2 = Ws @ Ws
    return {
        "eta2": e(S2),
        "zeta2": -e(O2),
        "s3": e(Ss @ Ss @ Ss),
        "so2": np.einsum("nij,njk,nki->n", Ss, Ws, Ws),
        "s2o2": np.einsum("nij,nij->n", S2, O2),
    }


def eval_subset(cases, names):
    """对给定基名子集做逐点 lstsq 反演. 返回全局相对残差/med·p90·max cond/最低有效点占比.

    p90/max cond: 跨 5 工况全部有效点的分位数 — 比 med cond 更能反映近壁病态
    (med cond 漂亮但尾部灾难, 见 RC2 诊断).
    """
    K = len(names)
    tot_sse = 0.0
    tot_ssb = 0.0
    conds = []
    all_pt_conds = []
    vfracs = []
    for fc in cases:
        b = fc["b6"]
        M = np.stack([fc["_b6"][n] for n in names], axis=2)  # (N,6,K)
        N = b.shape[0]
        valid = np.zeros(N, bool)
        pt_conds = []
        for i in range(N):
            s = np.linalg.svd(M[i], compute_uv=False)
            r = int(np.sum(s / max(s[0], 1e-300) > 1e-10))
            if r < K:
                continue
            valid[i] = True
            if s[-1] > 0:
                pt_conds.append(s[0] / s[-1])
        conds.append(float(np.median(pt_conds)) if pt_conds else np.inf)
        all_pt_conds.extend(pt_conds)
        vfracs.append(float(valid.mean()))
        # 残差用 min-norm lstsq (rcond 截断), 病态点不产生 1e26 级 a_m
        for i in range(N):
            if not valid[i]:
                continue
            sol, *_ = np.linalg.lstsq(M[i], b[i], rcond=1e-10)
            tot_sse += float(np.sum((M[i] @ sol - b[i]) ** 2))
        tot_ssb += float(np.sum(b[valid] ** 2))
    rel = float(np.sqrt(tot_sse / max(tot_ssb, 1e-300)))
    all_pt_conds = np.array(all_pt_conds) if all_pt_conds else np.array([np.inf])
    return dict(rel_resid=rel, med_cond=float(np.median(conds)),
                p90_cond=float(np.percentile(all_pt_conds, 90)),
                max_cond=float(np.max(all_pt_conds)),
                min_vfrac=float(np.min(vfracs)), med_cond_per_case=conds)


def pointwise_cond(basis6, names):
    """逐点条件数 (6xK SVD 奇异值比), 用于 valid_fit 掩码."""
    M = np.stack([basis6[n] for n in names], axis=2)
    N = M.shape[0]
    c = np.zeros(N)
    for i in range(N):
        s = np.linalg.svd(M[i], compute_uv=False)
        c[i] = s[0] / max(s[-1], 1e-300)
    return c


def fit_coverage(cases, names, cond_max, amax_cap):
    """换基段2 的 coverage 守门量: p90 只是代理, 这里算它代理的直接量.

    返回: nw_fit% (近壁 |y/h|>0.9 且过双掩码点占近壁物理有效点比例),
          all_fit% (全域同口径), amax_fit (fit 集上 max|a_m|).
    """
    nw_tot = nw_fit = all_fit = all_tot = 0
    amax = 0.0
    for fc in cases:
        a, _ = invert_case(fc["b6"], fc["_b6"], names)
        cnd = pointwise_cond(fc["_b6"], names)
        phys = (fc["k"] > 1e-4) & (fc["eps"] > 1e-12) \
            & np.isfinite(fc["b6"]).all(1) & np.isfinite(a).all(1)
        amax_row = np.nanmax(np.abs(a), axis=1) if a.size else np.zeros(len(phys))
        fitm = phys & (cnd <= cond_max) & (amax_row <= amax_cap)
        nw = np.abs(fc["yh"]) > 0.9
        nw_tot += int((nw & phys).sum()); nw_fit += int((nw & fitm).sum())
        all_tot += int(phys.sum()); all_fit += int(fitm.sum())
        if fitm.any():
            amax = max(amax, float(np.nanmax(np.abs(a[fitm]))))
    return dict(nw_fit_pct=100.0 * nw_fit / max(nw_tot, 1),
                all_fit_pct=100.0 * all_fit / max(all_tot, 1),
                amax_fit=amax)


def invert_case(b6, basis6, names):
    """逐点反演 a_m (仅 rank==K 的点有效). rcond 截断 → min-norm 解, a_m 有界."""
    K = len(names)
    M = np.stack([basis6[n] for n in names], axis=2)
    N = b6.shape[0]
    a = np.full((N, K), np.nan)
    valid = np.zeros(N, bool)
    for i in range(N):
        s = np.linalg.svd(M[i], compute_uv=False)
        if int(np.sum(s / max(s[0], 1e-300) > 1e-10)) < K:
            continue
        a[i], *_ = np.linalg.lstsq(M[i], b6[i], rcond=1e-10)
        valid[i] = True
    return a, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_pope2.npz")
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--min_vfrac", type=float, default=0.75)
    ap.add_argument("--cond_max", type=float, default=1e4,
                    help="valid_fit 掩码: 剔除逐点 cond 超过该值的点 (RC2 近壁病态)")
    ap.add_argument("--amax_cap", type=float, default=300.0,
                    help="valid_fit 掩码: 剔除 max_m|a_m| 超过该值的点 (反演幅值帽)")
    ap.add_argument("--p90_hard", type=float, default=0.7,
                    help="换基段1: p90 < p90_hard*base 才换 (现行行为)")
    ap.add_argument("--p90_slack", type=float, default=None,
                    help="换基段2 (放松): p90 < p90_slack*base 且 coverage 守门 "
                         "(nw_fit% 跌幅 <= nw_drop 且 max|a|<=amax_cap); None=关闭")
    ap.add_argument("--nw_drop", type=float, default=3.0,
                    help="段2 coverage 守门: 近壁 fit 覆盖率允许的最大跌幅 (百分点)")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    cases = []
    for ro in RO_CASES:
        fc = load_case(ro, variant="vort")  # Ot = Wbar (涡量)
        ke = (fc["k"] / fc["eps"])[:, None, None]
        Ss = ke * fc["S"]
        Wv = ke * fc["Ot"]                       # 涡量 (无帧旋转项)
        We = fc["Ot"].copy()                     # 扩展内禀: Wbar + 2*eps*Omega
        We[:, 1, 2] += 2.0 * fc["Om_eff"]
        We[:, 2, 1] -= 2.0 * fc["Om_eff"]
        We = ke * We
        Wg = fc["Ot"].copy()                     # Gatski-Speziale: Wbar + 1*eps*Omega
        Wg[:, 1, 2] += 1.0 * fc["Om_eff"]
        Wg[:, 2, 1] -= 1.0 * fc["Om_eff"]
        Wg = ke * Wg
        # 30 组候选 (T1 与 W 无关 -> 只保留一份)
        cand = []
        for n, T in pope_tensors(Ss, Wv):
            cand.append((n + "v", sym6(T)))
        for n, T in pope_tensors(Ss, We):
            if n == "T1":
                continue  # 与 T1v 完全相同
            cand.append((n + "e", sym6(T)))
        for n, T in pope_tensors(Ss, Wg):
            if n == "T1":
                continue  # 与 T1v 完全相同
            cand.append((n + "g", sym6(T)))
        # 把共享的 T1v 改名 T1
        cand = [("T1" if n == "T1v" else n, V) for n, V in cand]
        fc["_b6"] = dict(cand)
        # 不变量特征
        iv_v = invariants(Ss, Wv)
        iv_e = invariants(Ss, We)
        iv_g = invariants(Ss, Wg)
        fc["_feat"] = np.stack([
            iv_v["eta2"], iv_v["s3"],
            iv_v["zeta2"], iv_v["so2"], iv_v["s2o2"],
            iv_e["zeta2"], iv_e["so2"], iv_e["s2o2"],
            iv_g["zeta2"], iv_g["so2"], iv_g["s2o2"],
        ], axis=1)
        cases.append(fc)
        print(f"[pope] Ro={ro:>5}: {fc['checks']['n_pts']}pts, "
              f"zeta2v range [{iv_v['zeta2'].min():.3g},{iv_v['zeta2'].max():.3g}], "
              f"zeta2e range [{iv_e['zeta2'].min():.3g},{iv_e['zeta2'].max():.3g}], "
              f"zeta2g range [{iv_g['zeta2'].min():.3g},{iv_g['zeta2'].max():.3g}]")

    cand_names = [n for n, _ in cand]
    assert len(cand_names) == 28, f"候选基应为 28 个, 实得 {len(cand_names)}"

    # ---- 参照: 旧 ASM 固定 5 基 (= ext 版 T1/T2/T3/T4/T6) ----
    old = ["T1", "T2e", "T3e", "T4e", "T6e"]
    old_stats = eval_subset(cases, old)
    print(f"[pope] 旧基 (ext S/SOm/S2/Om2/Om2S): rel_resid={old_stats['rel_resid']:.2e} "
          f"med_cond={old_stats['med_cond']:.1f} min_vfrac={old_stats['min_vfrac']:.3f}")

    # ---- 贪心前向选择 ----
    sel = []
    greedy_trace = []
    prev_rel = None
    while len(sel) < args.kmax:
        best = None
        for n in cand_names:
            if n in sel:
                continue
            r = eval_subset(cases, sel + [n])
            if r["min_vfrac"] < args.min_vfrac:
                continue
            if best is None or r["rel_resid"] < best[1]["rel_resid"]:
                best = (n, r)
        if best is None:
            print("[pope] 贪心: 无可行候选 (逐点秩不足), 停止")
            break
        n, r = best
        greedy_trace.append(dict(step=len(sel) + 1, add=n,
                                 rel_resid=r["rel_resid"], med_cond=r["med_cond"],
                                 min_vfrac=r["min_vfrac"]))
        print(f"[pope] greedy step {len(sel)+1}: +{n:<5} rel_resid={r['rel_resid']:.2e} "
              f"med_cond={r['med_cond']:.1f} min_vfrac={r['min_vfrac']:.3f}")
        stop = False
        if r["rel_resid"] < 1e-5:
            stop = True  # 残差已达数值饱和
        elif prev_rel is not None and r["rel_resid"] > 0.95 * prev_rel:
            stop = True  # 无实质改进 (注意: 贪心加基残差单调不增, >0.95*prev 意味着几乎没降)
        sel.append(n)
        prev_rel = r["rel_resid"]
        if stop:
            break

    # ---- 条件数换基改善: 二段式 (v3 p90 放松方案) ----
    # 段1 (现行): rel_resid≤2× 且 p90 < p90_hard×base → 换
    # 段2 (--p90_slack N): p90 < N×base 但需 coverage 守门 (nw_fit%/幅值), 顺序在段1后
    # visited 集防振荡 (段2 换入高 p90 基后, 段1 可能又想换回去)
    def _swap_phase(sel, base, p90_gate, slack, visited):
        base_cov = fit_coverage(cases, sel, args.cond_max, args.amax_cap)
        passes = 0
        improved = True
        while improved and passes < 4:
            improved = False
            passes += 1
            for oi, oname in enumerate(sel):
                for inn in cand_names:
                    if inn in sel:
                        continue
                    trial = sel[:oi] + [inn] + sel[oi + 1:]
                    if tuple(trial) in visited:
                        continue
                    r = eval_subset(cases, trial)
                    if r["min_vfrac"] < args.min_vfrac:
                        continue
                    if r["rel_resid"] > max(2.0 * base["rel_resid"], 1e-6):
                        continue
                    tag = None
                    relaxed = False
                    if r["p90_cond"] < args.p90_hard * base["p90_cond"]:
                        tag = ""
                    elif slack is not None and r["p90_cond"] < slack * base["p90_cond"]:
                        cov = fit_coverage(cases, trial, args.cond_max, args.amax_cap)
                        if cov["nw_fit_pct"] >= base_cov["nw_fit_pct"] - args.nw_drop \
                                and cov["amax_fit"] <= args.amax_cap:
                            relaxed = True
                            tag = (f"  [relaxed: nw_fit {base_cov['nw_fit_pct']:.1f}%->"
                                   f"{cov['nw_fit_pct']:.1f}%, max|a| {cov['amax_fit']:.3g}]")
                        else:
                            print(f"[pope] relaxed-reject: {oname} -> {inn}  "
                                  f"p90 {r['p90_cond']:.1f} (<{slack:.0f}x base) 但 "
                                  f"nw_fit {base_cov['nw_fit_pct']:.1f}%->"
                                  f"{cov['nw_fit_pct']:.1f}% 跌超 {args.nw_drop}pp 或 "
                                  f"max|a| {cov['amax_fit']:.3g} 超帽")
                    if tag is not None:
                        print(f"[pope] swap{'[RELAXED]' if relaxed else ''}: "
                              f"{oname} -> {inn}  p90_cond {base['p90_cond']:.1f} -> "
                              f"{r['p90_cond']:.1f} (rel_resid "
                              f"{base['rel_resid']:.2e} -> {r['rel_resid']:.2e}){tag}")
                        sel = trial
                        base = r
                        base_cov = fit_coverage(cases, sel, args.cond_max, args.amax_cap)
                        visited.add(tuple(sel))
                        improved = True
                        break
                if improved:
                    break
        return sel, base

    visited = {tuple(sel)}
    base = eval_subset(cases, sel)
    sel, base = _swap_phase(sel, base, args.p90_hard, None, visited)   # 段1: 现行行为
    sel_s1 = list(sel)
    base_s1 = dict(base)
    if args.p90_slack is not None:
        print(f"[pope] === 段2 放松探索 (p90_slack={args.p90_slack:g}x, "
              f"nw_drop<={args.nw_drop:g}pp) ===")
        sel, base = _swap_phase(sel, base, args.p90_hard, args.p90_slack, visited)
        if tuple(sel) != tuple(sel_s1):
            print(f"[pope] 段2 换基生效: {sel_s1} -> {sel}")
    print(f"[pope] 最终基底 ({len(sel)}): {sel}")
    print(f"[pope] rel_resid={base['rel_resid']:.2e} med_cond={base['med_cond']:.1f} "
          f"p90_cond={base['p90_cond']:.1f} max_cond={base['max_cond']:.3g} "
          f"min_vfrac={base['min_vfrac']:.3f}")

    # ---- 候选子集对比表 (RC3: 评估 T6v/T10v 替换 T5v, vw 支撑量纲更良态) ----
    variants = [("selected", sel)]
    if "T5v" in sel:
        for repl in ("T6v", "T10v", "T9v"):
            variants.append((f"T5v->{repl}", [repl if n == "T5v" else n for n in sel]))
    print(f"\n[pope] === 候选子集对比 (cond_max={args.cond_max:g}, "
          f"amax_cap={args.amax_cap:g}) ===")
    print(f"{'subset':<12} {'rel_resid':>10} {'med_cond':>9} {'p90_cond':>10} "
          f"{'min_vfrac':>10} {'max|a|@fit':>11} {'nw_fit%':>8}")
    subset_stats = {}
    for vname, vs in variants:
        r = eval_subset(cases, vs)
        amax_all = 0.0
        nw_tot = nw_fit = 0
        for fc in cases:
            a, _ = invert_case(fc["b6"], fc["_b6"], vs)
            cnd = pointwise_cond(fc["_b6"], vs)
            phys = (fc["k"] > 1e-4) & (fc["eps"] > 1e-12) \
                & np.isfinite(fc["b6"]).all(1) & np.isfinite(a).all(1)
            fitm = phys & (cnd <= args.cond_max) \
                & (np.nanmax(np.abs(a), axis=1) <= args.amax_cap)
            if fitm.any():
                amax_all = max(amax_all, float(np.nanmax(np.abs(a[fitm]))))
            nw = np.abs(fc["yh"]) > 0.9
            nw_tot += int((nw & phys).sum()); nw_fit += int((nw & fitm).sum())
        nw_pct = 100.0 * nw_fit / max(nw_tot, 1)
        subset_stats[vname] = dict(basis=vs, rel_resid=r["rel_resid"],
                                   p90_cond=r["p90_cond"], max_amax=amax_all,
                                   nw_fit_pct=nw_pct)
        print(f"{vname:<12} {r['rel_resid']:>10.2e} {r['med_cond']:>9.1f} "
              f"{r['p90_cond']:>10.1f} {r['min_vfrac']:>10.3f} "
              f"{amax_all:>11.3g} {nw_pct:>7.1f}%")
    print()

    # ---- 反演 + 保存 ----
    save = {"sel_basis": np.array(sel), "featB_keys": np.array(FEAT),
            "featA_keys": np.array(FEAT), "cand_basis": np.array(cand_names)}
    report = {"selected_basis": sel, "greedy_trace": greedy_trace,
              "stage1_basis": sel_s1,
              "final_stats": {k: base[k] for k in ("rel_resid", "med_cond",
                                                   "p90_cond", "max_cond", "min_vfrac")},
              "valid_fit_rule": f"cond<={args.cond_max:g} & max|a_m|<={args.amax_cap:g}",
              "subset_comparison": subset_stats,
              "old_ext5_stats": {k: old_stats[k] for k in ("rel_resid", "med_cond", "min_vfrac")},
              "pope_ref": "Pope, S.B. (1975) JFM 72:331-340; basis T1..T10; "
                          "invariants {S^2},{W^2},{S^3},{SW^2},{S^2W^2}",
              "w_variants": {"v": "Wbar (vorticity)",
                             "e": "Omega_tilde = Wbar + 2*eps*Om (ext, 黄/马 2008 式(19))",
                             "g": "W* = Wbar + 1*eps*Om (Gatski & Speziale 1993 JFM 254:59)"}}
    for i, fc in enumerate(cases):
        p = f"c{i}_"
        a, rankv = invert_case(fc["b6"], fc["_b6"], sel)
        valid = rankv & (fc["k"] > 1e-4) & (fc["eps"] > 1e-12) \
                & np.isfinite(fc["b6"]).all(1) & np.isfinite(a).all(1)
        # valid_fit: 在 valid 上再剔病态点 (RC2) — 只影响 SR 训练目标,
        # eval 仍用完整 valid 诚实度量近壁外推
        cnd = pointwise_cond(fc["_b6"], sel)
        amax_row = np.nanmax(np.abs(a), axis=1) if a.size else np.zeros(len(valid))
        valid_fit = valid & (cnd <= args.cond_max) & (amax_row <= args.amax_cap)
        nw = np.abs(fc["yh"]) > 0.9
        save.update({
            p + "ro": fc["ro"], p + "yh": fc["yh"], p + "yp": fc["yp"],
            p + "k": fc["k"], p + "eps": fc["eps"], p + "valid": valid,
            p + "valid_fit": valid_fit, p + "cond": cnd,
            p + "b6": fc["b6"],
            p + "R6": np.stack([fc["R"][c] for c in COMP6], 1),
            p + "featB": fc["_feat"], p + "featA": fc["_feat"],
            p + "a": a, p + "invres": np.full(valid.shape, np.nan),
        })
        for n, V in fc["_b6"].items():
            save[f"{p}B_{n}"] = V
        for mi, n in enumerate(sel):
            save[f"{p}T_{n}"] = fc["_b6"][n]
            # 逐点权重: ||T_m||_F^2 (6 分量平方和) — 应力空间 fitness
            save[f"{p}W_{n}"] = np.sum(fc["_b6"][n] ** 2, axis=1)
        # 反演相对残差 (仅有效点有值)
        tmp = np.full(valid.shape, np.nan)
        if valid.any():
            Mv = np.stack([fc["_b6"][n] for n in sel], 2)[valid]   # (Nv,6,K)
            bhat = np.einsum("nmk,nk->nm", Mv, a[valid])
            tmp[valid] = (np.linalg.norm(bhat - fc["b6"][valid], axis=1)
                          / np.maximum(np.linalg.norm(fc["b6"][valid], axis=1), 1e-12))
        save[p + "invres"] = tmp
        print(f"[pope] Ro={fc['ro']:>5}: valid {int(valid.sum())}/{len(valid)}, "
              f"valid_fit {int(valid_fit.sum())} (近壁|y/h|>0.9: "
              f"{int((nw & valid).sum())}→{int((nw & valid_fit).sum())}), "
              f"max|a|@fit={np.nanmax(np.abs(a[valid_fit])):.3g}, "
              f"反演残差 med={np.nanmedian(tmp):.2e}")
    np.savez_compressed(os.path.join(OUT_DIR, args.out), **save)
    with open(os.path.join(OUT_DIR, "prep_report_pope.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[pope] -> {os.path.join(OUT_DIR, args.out)}")
    _plots(cases, sel)


def _plots(cases, sel):
    """a_m(y) 反演剖面 + 表达性校核图 (与旧 prep 同风格)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 反演 a_m (与保存逻辑一致)
    prof = {}
    for fc in cases:
        a, valid = invert_case(fc["b6"], fc["_b6"], sel)
        prof[fc["ro"]] = (a, valid)
    fig, axes = plt.subplots(1, len(sel), figsize=(2.9 * len(sel), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    cols = plt.cm.viridis(np.linspace(0, 1, len(cases)))
    for mi, n in enumerate(sel):
        for j, fc in enumerate(cases):
            a, valid = prof[fc["ro"]]
            axes[mi].plot(a[:, mi], fc["yh"], color=cols[j],
                          label=f"$Ro_\\tau$={fc['ro']:g}")
        axes[mi].set_title(f"$a_{{{n}}}$")
        axes[mi].grid(alpha=0.3); axes[mi].axvline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("$y/h$")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prep_pope_am_profiles.png"), dpi=130)
    plt.close(fig)
    print("[pope] figures -> prep_pope_am_profiles.png")


if __name__ == "__main__":
    main()
