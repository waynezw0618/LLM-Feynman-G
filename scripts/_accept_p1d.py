# -*- coding: utf-8 -*-
"""P1d 验收: 改进管线 (A1/A2/A3 + B1/B2/B3) 后的 4 臂对比.

臂:
  heuristic_gsonly        (P1c 基线, 旧管线)
  minimax_pope_gsonly     (P1c,     旧管线)
  heuristic_gsonly_v2     (P1d,     新管线, forbid T5g:sub,div)
  minimax_pope_gsonly_v2  (P1d,     新管线, forbid T5g:sub,div)

V1: 5 目标 stress loss (+ yw 占用 + 复杂度)
V2: 逐 Ro NMSE (best.metrics, a_m 系数空间加权)
V3: 全量重构 (theta 正确分块): 近壁 |y/h|>0.9 6 分量 NMSE + vw 中心峰 |y/h|<0.15
V4: T5g 公式结构检查 (无 sub/div, B3 验证)
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:/AI/LLM-Feynman-G")

ASMOUT = r"D:/AI/LLM-Feynman-G/outputs/asm"
RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
TARGETS = ["T1", "T2v", "T3v", "T4v", "T5g"]
ARMS = [("heuristic_gsonly", "heur"),
        ("minimax_pope_gsonly", "minx"),
        ("heuristic_gsonly_v2", "h_v2"),
        ("minimax_pope_gsonly_v2", "m_v2")]

from lgf.compiler import CompiledGraph
from lgf.dsl import parse_dsl

zh = np.load(f"{ASMOUT}/dataset_pope_gsonly.npz")
zg = np.load(f"{ASMOUT}/dataset_pope3_wy.npz")   # B_T5g 兜底 (P1c 写盘 bug)


def basis_of(i, n):
    p = f"c{i}_"
    if n == "T5g" and (p + "B_T5g") not in zh.files:
        return zg[p + "B_T5g"]
    return zh[p + f"B_{n}"]


def load_arm(tag):
    s = json.load(open(f"{ASMOUT}/runs_{tag}/summary_B.json", encoding="utf-8"))
    return s


def a_full_for_arm(s, i):
    """正确 theta 分块: [globals, case_i params] → 5 目标 a_m (N,5)."""
    p = f"c{i}_"
    featA = zh[p + "featA"]
    out = np.zeros((featA.shape[0], 5))
    for ti, t in enumerate(TARGETS):
        best = s["targets"][t]
        try:
            g = parse_dsl(best["dsl"], var_names=s["var_names"])
            cg = CompiledGraph(g, backend="numpy")
            n_g, n_c = cg.n_g, cg.n_c
            th = np.asarray(best["theta"], dtype=float)
            tl = np.concatenate([th[:n_g],
                                 th[n_g + i * n_c: n_g + (i + 1) * n_c]])
            out[:, ti] = cg.eval_case(tl, featA)
        except Exception as e:
            print(f"  [WARN] {s['tag']}/{t}/c{i} eval fail: {e}")
    return out


# ============ V1: 5 目标 loss ============
print("=" * 78)
print("P1d 验收: 4 臂对比 (旧管线 heur/minx vs 新管线 h_v2/m_v2)")
print("=" * 78)
S = {tag: load_arm(tag) for tag, _ in ARMS}
print("\n## V1: 5 目标 stress loss (+yw占用+节点数)")
hdr = f"  {'T':<5}" + "".join(f"{lab:>10}" for _, lab in ARMS) + "   yw(4臂)   nodes(4臂)"
print(hdr)
means = {lab: [] for _, lab in ARMS}
for t in TARGETS:
    row = f"  {t:<5}"
    yws, nodes = [], []
    for tag, lab in ARMS:
        b = S[tag]["targets"][t]
        means[lab].append(b["loss"])
        row += f"{b['loss']:>10.4f}"
        yws.append("Y" if "yw" in b["dsl"] else "N")
        nodes.append(b["complexity"]["nodes"])
    row += "   " + "".join(yws) + "     " + "/".join(str(n) for n in nodes)
    print(row)
row = "  mean "
for _, lab in ARMS:
    row += f"{np.mean(means[lab]):>10.4f}"
print(row)

# ============ V2: 逐 Ro NMSE ============
print("\n## V2: 逐 Ro NMSE (a_m 加权系数空间; 重点 T1/T2v/T5g)")
for t in ["T1", "T2v", "T5g"]:
    print(f"  --- {t} ---")
    print(f"  {'arm':<6}", "  ".join(f"Ro{ro:<6g}" for ro in RO_CASES), "  worst")
    for tag, lab in ARMS:
        ms = S[tag]["targets"][t]["metrics"]
        vals = [ms.get(f"Ro{ro:g}_nmse", np.nan) for ro in RO_CASES]
        print(f"  {lab:<6}", "  ".join(f"{v:7.3f}" for v in vals),
              f"  {np.nanmax(vals):7.3f}")

# ============ V3: 全量重构 ============
print("\n## V3a: 近壁 (|y/h|>0.9, valid) 6 分量 NMSE — 5 Ro 均值")
COMP = ["uu", "vv", "ww", "uv", "uw", "vw"]
NW = {}
for tag, lab in ARMS:
    nw_acc = {c: [] for c in COMP}
    for i in range(5):
        p = f"c{i}_"
        a = a_full_for_arm(S[tag], i)
        M = np.stack([basis_of(i, n) for n in TARGETS], axis=2)  # (N,6,5)
        b_hat = np.einsum("nkm,nm->nk", M, a)
        b6 = zh[p + "b6"]; yh = zh[p + "yh"]; v = zh[p + "valid"]
        m = (np.abs(yh) > 0.9) & v & np.isfinite(b_hat).all(1)
        for ci, c in enumerate(COMP):
            den = max(float(np.nanvar(b6[m, ci])), 1e-12)
            nw_acc[c].append(float(np.mean((b_hat[m, ci] - b6[m, ci]) ** 2) / den))
    NW[tag] = {c: float(np.mean(v)) for c, v in nw_acc.items()}
print(f"  {'arm':<6}", "  ".join(f"{c:>8}" for c in COMP))
for tag, lab in ARMS:
    print(f"  {lab:<6}", "  ".join(f"{NW[tag][c]:8.3f}" for c in COMP))

print("\n## V3b: vw 中心峰 (|y/h|<0.15, valid) b_vw 均值 (DNS 参考各 Ro)")
print(f"  {'Ro':<6}", "  ".join(f"{lab:>7}" for _, lab in ARMS), "   DNS    达标(4臂)")
DNS_REF = {7.5: 0.037, 15: 0.045, 30: 0.059, 75: 0.054, 150: 0.034}
pk = {lab: 0 for _, lab in ARMS}
for i, ro in enumerate(RO_CASES):
    p = f"c{i}_"
    b6 = zh[p + "b6"]; yh = zh[p + "yh"]; v = zh[p + "valid"]
    cn = (np.abs(yh) < 0.15) & v
    row = f"  Ro{ro:<4g}"
    for tag, lab in ARMS:
        a = a_full_for_arm(S[tag], i)
        M = np.stack([basis_of(i, n) for n in TARGETS], axis=2)
        b_hat = np.einsum("nkm,nm->nk", M, a)
        vw = float(np.nanmean(b_hat[cn, 5]))
        if vw > 0.03:
            pk[lab] += 1
        row += f"{vw:8.3f}"
    print(row, f"  {DNS_REF[ro]:.3f}")
print("  达标数:", "  ".join(f"{lab}:{pk[lab]}/5" for _, lab in ARMS))

# ============ V4: T5g 结构检查 ============
print("\n## V4: T5g 公式结构 (B3 验证: v2 臂不得含 sub/div)")
for tag, lab in ARMS:
    d = S[tag]["targets"]["T5g"]["dsl"]
    has_bad = ("-" in d.replace("^-", "") and " sub(" in d) or "/" in d
    # 节点级检查更可靠
    try:
        g = parse_dsl(d, var_names=S[tag]["var_names"])
        from lgf.graph import walk
        ops = {n.op for _, n in walk(g.root)}
    except Exception:
        ops = {"?"}
    bad = sorted(ops & {"sub", "div"})
    print(f"  {lab:<6} ops={sorted(ops)}  {'❌ 含禁用算子 ' + str(bad) if bad else '✅'}")
    print(f"         dsl: {d[:110]}")

# ============ 结论 ============
print("\n" + "=" * 78)
print("结论摘要")
print("=" * 78)
ml = {lab: np.mean(means[lab]) for _, lab in ARMS}
print(f"  5 目标 loss 均值:  " + "  ".join(f"{lab}={ml[lab]:.4f}" for _, lab in ARMS))
best_arm = min(ml, key=ml.get)
print(f"  最优臂: {best_arm}")
for a, b in [("m_v2", "minx"), ("m_v2", "h_v2"), ("m_v2", "heur")]:
    d = (ml[a] - ml[b]) / ml[b] * 100
    print(f"  {a} vs {b}: {d:+.1f}%")
print(f"  vw 中心峰: " + "  ".join(f"{lab}={pk[lab]}/5" for _, lab in ARMS))
