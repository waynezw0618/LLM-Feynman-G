"""P1c 验收: minimax x pope_gsonly vs heuristic x pope_gsonly.

V1: 近壁 (|y/h|>0.9) 6 分量 NMSE  (按 rc_pope_accept.py 口径, valid 全点含外推)
V2: vw 中心峰 (|y/h|<0.15) 重构 b_vw 均值 > 0.03  (DNS 0.037-0.059)
V3: yw 占用 (best 公式含 yw)
V4: 反演健康度 max|a|@fit < 1e3
V5: 5 目标 stress loss 均值
"""
import json
import numpy as np
import os
import sys
sys.path.insert(0, r"D:/AI/LLM-Feynman-G")
sys.path.insert(0, r"D:/AI/LLM-Feynman-G/scripts")

from rc_asm_prep import load_case
from rc_pope_prep import load_pope

ASMOUT = r"D:/AI/LLM-Feynman-G/outputs/asm"
RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
TARGETS = ["T1", "T2v", "T3v", "T4v", "T5g"]


def load_best(tag, t):
    return json.load(open(f"{ASMOUT}/runs_{tag}/B_a_{t}_final.json"))["best"]


def get_a_full(tag, t, ds_name="dataset_pope_gsonly.npz"):
    """重构 6 分量 b_hat 在 5 Ro 上的近壁 / 中心峰结果."""
    fc = load_pope(ds_name)
    b6 = fc["b6"]
    yh = fc["yh"]
    valid = fc["valid"]
    valid_fit = fc["valid_fit"]
    k = fc["k"]
    bsel = [fc["basis6"][n] for n in TARGETS]  # 5 基
    a_full = np.full((len(b6), 5), np.nan)
    M = np.stack(bsel, axis=2)
    for i in range(len(b6)):
        if not valid_fit[i]:
            continue
        try:
            a_full[i], *_ = np.linalg.lstsq(M[i], b6[i], rcond=1e-10)
        except Exception:
            pass
    b_hat = np.einsum("nkm,nk->nm", np.where(np.isfinite(a_full)[:, None, :], M, 0),
                      np.where(np.isfinite(a_full), a_full, 0))
    # 仅 valid 的点用 a
    b_hat_v = np.where(valid[:, None], b_hat, np.nan)
    return b_hat_v, yh, valid, valid_fit, a_full


def fit6(tag, t):
    """返回 (b_hat[6], yh, valid, valid_fit, a_full)."""
    a, y, v, vf, af = get_a_full(tag, t)
    return a, y, v, vf, af


def get_actual_recon(tag, t):
    """用每个 (Ro, target) 的 best 公式实际在 5 Ro 上算 b_hat (用 compile_paramfit)."""
    from lgf import paramfit
    best = load_best(tag, t)
    dsl = best["dsl"]
    theta = best["theta"]
    # 变量名顺序: rc_asm_run.py 用 ds[featB_keys]
    # 改用直接读 npz 的 featA/featB 顺序
    # 但编译要 var_names 一致 — 简化: 用 metric 中已算好的
    return best["metrics"]


# ============= 主表 =============
print("=" * 80)
print("P1c 验收: minimax x pope_gsonly vs heuristic x pope_gsonly (单因子)")
print("=" * 80)

# --- 1) 5 目标 loss 对比 ---
print("\n## 1. 5 目标 stress loss 对比 (越小越好)")
print(f"  {'T':<4}  {'heuristic':>10}  {'minimax':>10}  {'delta':>9}  {'yw_h':>5}  {'yw_m':>5}")
losses_h, losses_m = [], []
for t in TARGETS:
    h = load_best("heuristic_gsonly", t)
    m = load_best("minimax_pope_gsonly", t)
    losses_h.append(h["loss"])
    losses_m.append(m["loss"])
    d = m["loss"] - h["loss"]
    s = "+" if d >= 0 else ""
    yw_h = "✓" if "yw" in h["dsl"] else "✗"
    yw_m = "✓" if "yw" in m["dsl"] else "✗"
    print(f"  {t:<4}  {h['loss']:>10.4f}  {m['loss']:>10.4f}  {s}{d:>8.4f}  {yw_h:>5}  {yw_m:>5}")

mh = np.mean(losses_h)
mm = np.mean(losses_m)
d = mm - mh
s = "+" if d >= 0 else ""
print(f"  {'mean':<4}  {mh:>10.4f}  {mm:>10.4f}  {s}{d:>8.4f}")

# --- 2) V1: 近壁 (|y/h|>0.9) 6 分量 NMSE ---
print("\n## 2. V1: 近壁 (|y/h|>0.9) 6 分量 NMSE  — valid 全点 (含外推区, 诚实口径)")
# 重构 b_hat 然后按 valid 算
print("  重新跑重构... (慢)")

# 跳过慢重构: 直接用 best['metrics'] 里的 Ro NMSE (那是全点 valid 的 NMSE)
# 近壁 |y/h|>0.9 需手动算.  简化: 用 metrics 里的 R^2 反推 NMSE — 但 NMSE 是 stress-weighted 全点.
# 真正近壁需重构 b_hat.  写个 helper.
import importlib
sys.path.insert(0, r"D:/AI/LLM-Feynman-G")
from lgf import paramfit as pf

ASMOUT_R = r"D:/AI/LLM-Feynman-G/outputs/asm"
TAG_H, TAG_M = "heuristic_gsonly", "minimax_pope_gsonly"

# 对每个 target, 编译 best 公式然后重构 6 分量
def reconstruct_one(tag, t):
    best = load_best(tag, t)
    dsl = best["dsl"]
    theta = best["theta"]
    # 读 var_names from summary
    summary = json.load(open(f"{ASMOUT_R}/runs_{tag}/summary_B.json"))
    var_names = summary["var_names"]
    # 编译
    fn = pf.compile_graph(dsl, var_names=var_names)
    # 加载 gsonly npz, 用 featA 重构
    z = np.load(f"{ASMOUT_R}/dataset_pope_gsonly.npz")
    yh_all, k_all, b6_all, valid_all = [], [], [], []
    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        featA = z[p + "featA"]
        try:
            a_pred = fn(featA, *theta)  # 5 dims = a_m
        except Exception as e:
            print(f"  [WARN] {tag}/{t}/Ro{ro} eval fail: {e}")
            a_pred = np.zeros((len(featA), 5))
        bsel = [z[p + f"B_{n}"] for n in TARGETS]
        M = np.stack(bsel, axis=2)
        b_hat = np.einsum("nkm,nk->nm", M, a_pred)
        b6 = z[p + "b6"]
        yh = z[p + "yh"]
        valid = z[p + "valid"]
        yh_all.append(yh); k_all.append(None)
        b6_all.append((b_hat, b6))
        valid_all.append(valid)
    return yh_all, b6_all, valid_all


def near_wall_nmse(tag, t, comp_idx):
    yh_all, b6_all, valid_all = reconstruct_one(tag, t)
    errs = []
    for i in range(len(RO_CASES)):
        yh = yh_all[i]
        b_hat, b6 = b6_all[i]
        v = valid_all[i]
        nw = (np.abs(yh) > 0.9) & v
        if nw.sum() == 0:
            continue
        denom = np.nanmaximum(np.nanvar(b6[nw, comp_idx]), 1e-12)
        errs.append(((b_hat[nw, comp_idx] - b6[nw, comp_idx]) ** 2).mean() / denom)
    return np.mean(errs) if errs else np.nan


COMP = ["uu", "vv", "ww", "uv", "uw", "vw"]
print(f"\n  {'T':<4}  comp  heur      minx     delta")
v1_h, v1_m = [], []
for t in TARGETS:
    print(f"  {t}")
    for ci, c in enumerate(COMP):
        nh = near_wall_nmse(TAG_H, t, ci)
        nm = near_wall_nmse(TAG_M, t, ci)
        d = nm - nh
        s = "+" if d >= 0 else ""
        marker = ""
        if not np.isnan(nh) and not np.isnan(nm):
            print(f"           {c:<3}  {nh:>7.4f}  {nm:>7.4f}  {s}{d:>6.4f}")
    # V1 分数: 6 分量 < 0.3 的个数
    pass

# --- 3) V2: vw 中心峰 (T5g 公式 → 重构 b_vw 在 |y/h|<0.15) ---
print("\n## 3. V2: vw 中心峰 (|y/h|<0.15) 重构 b_vw 均值 (T5g 公式驱动)")
print(f"  {'Ro':<5}  DNS     heur    minx    heur_✓  minx_✓")
DNS_REF = {7.5: 0.037, 15: 0.045, 30: 0.059, 75: 0.054, 150: 0.034}
for i, ro in enumerate(RO_CASES):
    yh_all_h, b6_all_h, valid_all_h = reconstruct_one(TAG_H, "T5g")
    yh_all_m, b6_all_m, valid_all_m = reconstruct_one(TAG_M, "T5g")
    b_h, b6_h = b6_all_h[i]
    b_m, b6_m = b6_all_m[i]
    yh = yh_all_h[i]
    cn = (np.abs(yh) < 0.15)
    vw_dns = b6_h[cn, 5].mean()
    vw_h = b_h[cn, 5].mean()
    vw_m = b_m[cn, 5].mean()
    h_pass = "✓" if vw_h > 0.03 else "✗"
    m_pass = "✓" if vw_m > 0.03 else "✗"
    print(f"  Ro{ro:<4g}  {vw_dns:.3f}   {vw_h:.3f}   {vw_m:.3f}    {h_pass}       {m_pass}")

# --- 4) V3: yw 占用 ---
print("\n## 4. V3: yw 占用 (best 公式含 yw)")
h_yw, m_yw = 0, 0
for t in TARGETS:
    h = load_best(TAG_H, t)
    m = load_best(TAG_M, t)
    if "yw" in h["dsl"]: h_yw += 1
    if "yw" in m["dsl"]: m_yw += 1
print(f"  heuristic: {h_yw}/5, minimax: {m_yw}/5")

# --- 5) V4: 反演健康度 max|a|@fit ---
print("\n## 5. V4: 反演健康度 max|a|@fit")
z = np.load(f"{ASMOUT_R}/dataset_pope_gsonly.npz")
bsel = [z[f"c0_B_{n}"] for n in TARGETS]
amax_all = []
for i in range(5):
    p = f"c{i}_"
    valid_fit = z[p + "valid_fit"]
    if valid_fit.sum() == 0:
        continue
    amax_all.append(np.nanmax(np.abs(z[p + "a"][valid_fit])))
print(f"  max across 5 Ro: {max(amax_all):.3g}  (要求 < 1e3)")
