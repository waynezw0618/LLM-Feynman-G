"""P1c 验收 v2: 轻量版, 直接从 npz 读不依赖 rc_pope_prep."""
import json
import numpy as np
import os
import sys
sys.path.insert(0, r"D:/AI/LLM-Feynman-G")
sys.path.insert(0, r"D:/AI/LLM-Feynman-G/scripts")

from lgf.compiler import CompiledGraph
from lgf.dsl import parse_dsl

ASMOUT = r"D:/AI/LLM-Feynman-G/outputs/asm"
RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
TARGETS = ["T1", "T2v", "T3v", "T4v", "T5g"]
COMP = ["uu", "vv", "ww", "uv", "uw", "vw"]
TAG_H, TAG_M = "heuristic_gsonly", "minimax_pope_gsonly"


def load_best(tag, t):
    return json.load(open(f"{ASMOUT}/runs_{tag}/B_a_{t}_final.json"))["best"]


def reconstruct(tag, t, ds_name="dataset_pope_gsonly.npz"):
    """对 5 Ro 重构 b_hat (6 分量).
    流程: 对 5 个 target (T1..T5g) 各编译 + eval_case 得到 5 个 a_m (N,), 再 b_hat = Σ a_m T_m.
    """
    summary_h = json.load(open(f"{ASMOUT}/runs_{TAG_H}/summary_B.json"))
    summary_m = json.load(open(f"{ASMOUT}/runs_{TAG_M}/summary_B.json"))
    var_names_h = summary_h["var_names"]
    var_names_m = summary_m["var_names"]
    z = np.load(f"{ASMOUT}/{ds_name}")
    z_g = np.load(f"{ASMOUT}/dataset_pope3_wy.npz")  # T5g 来源 (gsonly npz 未保存 B_T5g)
    out = []
    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        featA = z[p + "featA"]
        bsel = []
        for n in TARGETS:
            if n == "T5g":
                # gsonly npz 不含 B_T5g, 从 dataset_pope3_wy 取
                if (p + "B_T5g") in z_g.files:
                    bsel.append(z_g[p + "B_T5g"])
                else:
                    # 兜底: 用 gsonly 自己的 B_T5v 代替 (只是验收脚本, 不影响 GP 训练)
                    bsel.append(z[p + "B_T5v"])
            else:
                bsel.append(z[p + f"B_{n}"])
        M = np.stack(bsel, axis=1)  # (N, 5, 6)  — 5 个基 × 6 分量
        a_all = np.zeros((len(featA), 5))
        for mi, n in enumerate(TARGETS):
            best = load_best(tag, n)
            dsl = best["dsl"]
            vn = var_names_h if n in ["T1", "T2v", "T3v", "T4v", "T5v"] else var_names_m
            g = parse_dsl(dsl, var_names=vn)
            cg = CompiledGraph(g, backend="numpy")
            n_g = cg.n_g; n_c = cg.n_c
            tl = np.concatenate([best["theta"], np.zeros(n_c)])
            try:
                a_all[:, mi] = cg.eval_case(tl, featA)
            except Exception:
                a_all[:, mi] = 0.0
        b_hat = np.einsum("nkm,nk->nm", M, a_all)
        b6 = z[p + "b6"]
        yh = z[p + "yh"]
        valid = z[p + "valid"]
        out.append({"ro": ro, "b_hat": b_hat, "b6": b6, "yh": yh, "valid": valid})
    return out


def near_wall_nmse(recon, comp_idx):
    errs = []
    for r in recon:
        nw = (np.abs(r["yh"]) > 0.9) & r["valid"]
        if nw.sum() == 0: continue
        denom = np.maximum(np.nanvar(r["b6"][nw, comp_idx]), 1e-12)
        e = ((r["b_hat"][nw, comp_idx] - r["b6"][nw, comp_idx]) ** 2).mean() / denom
        errs.append(e)
    return float(np.mean(errs)) if errs else float("nan")


# 1) loss
print("=" * 80)
print("P1c 验收: minimax x pope_gsonly vs heuristic x pope_gsonly (单因子)")
print("=" * 80)
print("\n## 1. 5 目标 stress loss 对比 (越小越好)")
print(f"  {'T':<4}  {'heuristic':>10}  {'minimax':>10}  {'delta':>10}  yw_h  yw_m  n_h  n_m")
losses_h, losses_m = [], []
for t in TARGETS:
    h = load_best(TAG_H, t); m = load_best(TAG_M, t)
    losses_h.append(h["loss"]); losses_m.append(m["loss"])
    d = m["loss"] - h["loss"]
    s = "+" if d >= 0 else ""
    yw_h = "Y" if "yw" in h["dsl"] else "N"
    yw_m = "Y" if "yw" in m["dsl"] else "N"
    print(f"  {t:<4}  {h['loss']:>10.4f}  {m['loss']:>10.4f}  {s}{d:>9.4f}  {yw_h:>4}  {yw_m:>4}  {h['complexity']['nodes']:>3}  {m['complexity']['nodes']:>3}")
mh = np.mean(losses_h); mm = np.mean(losses_m)
print(f"  mean  {mh:>10.4f}  {mm:>10.4f}  {mm-mh:+10.4f}")

# 2) V1
print("\n## 2. V1: 近壁 (|y/h|>0.9) 6 分量 NMSE  — valid 全点 (含外推区)")
v1_table = {}
for t in TARGETS:
    print(f"  {t}:")
    rh = reconstruct(TAG_H, t); rm = reconstruct(TAG_M, t)
    for ci, c in enumerate(COMP):
        nh = near_wall_nmse(rh, ci)
        nm = near_wall_nmse(rm, ci)
        v1_table[(t, c)] = (nh, nm)
        d = nm - nh
        s = "+" if d >= 0 else ""
        print(f"    {c:<3}  heur={nh:>7.4f}  minx={nm:>7.4f}  d={s}{d:>6.4f}")
print("\n  V1 分数 (6 分量 NMSE<0.3):")
for tag, tn in [(TAG_H, "heuristic"), (TAG_M, "minimax")]:
    pc = sum(1 for (t,c),(nh,nm) in v1_table.items() if (nh if tag==TAG_H else nm) < 0.3 and not np.isnan(nh if tag==TAG_H else nm))
    tot = sum(1 for (t,c),(nh,nm) in v1_table.items() if not np.isnan(nh if tag==TAG_H else nm))
    print(f"    {tn:<10} {pc}/{tot}")

# 3) V2
print("\n## 3. V2: vw 中心峰 (|y/h|<0.15, T5g 公式驱动)")
print(f"  {'Ro':<6}  DNS   heur  minx  h  m")
rh = reconstruct(TAG_H, "T5g"); rm = reconstruct(TAG_M, "T5g")
ph, pm = 0, 0
for r_h, r_m in zip(rh, rm):
    cn = np.abs(r_h["yh"]) < 0.15
    dns = r_h["b6"][cn, 5].mean()
    vh = r_h["b_hat"][cn, 5].mean()
    vm = r_m["b_hat"][cn, 5].mean()
    hp = "Y" if vh > 0.03 else "N"
    mp = "Y" if vm > 0.03 else "N"
    if vh > 0.03: ph += 1
    if vm > 0.03: pm += 1
    print(f"  Ro{r_h['ro']:<5g}  {dns:.3f}  {vh:.3f}  {vm:.3f}  {hp}   {mp}")
print(f"  vw 中心峰达标: heur {ph}/5, minx {pm}/5")

# 4) V3
print("\n## 4. V3: yw 占用 (best 公式含 yw)")
hy = sum(1 for t in TARGETS if "yw" in load_best(TAG_H, t)["dsl"])
my = sum(1 for t in TARGETS if "yw" in load_best(TAG_M, t)["dsl"])
print(f"  heuristic: {hy}/5, minimax: {my}/5")

# 5) V4
print("\n## 5. V4: 反演健康度 max|a|@fit")
z = np.load(f"{ASMOUT}/dataset_pope_gsonly.npz")
amax_all = []
for i in range(5):
    p = f"c{i}_"
    vf = z[p + "valid_fit"]
    if vf.sum() == 0: continue
    amax_all.append(float(np.nanmax(np.abs(z[p + "a"][vf]))))
print(f"  max across 5 Ro: {max(amax_all):.3g}  (要求 < 1e3)")

# 6) 结论
print("\n" + "=" * 80)
print("结论摘要")
print("=" * 80)
print(f"  5 目标 loss 均值:  heur={mh:.4f}  minx={mm:.4f}  d={mm-mh:+.4f}")
print(f"  vw 中心峰 5/5:    heur={ph}/5   minx={pm}/5")
print(f"  yw 占用 5/5:      heur={hy}/5   minx={my}/5")
