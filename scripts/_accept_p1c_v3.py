"""P1c V1 简化版: 直接读 best metrics, 不重构 b_hat (避免 RuntimeWarning 中断)."""
import json, os, sys, numpy as np
sys.path.insert(0, r"D:/AI/LLM-Feynman-G")
import warnings
warnings.filterwarnings("ignore")

ASMOUT = r"D:/AI/LLM-Feynman-G/outputs/asm"
RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
TARGETS = ["T1", "T2v", "T3v", "T4v", "T5g"]
TAG_H, TAG_M = "heuristic_gsonly", "minimax_pope_gsonly"

# --- 5 目标 loss (从 summary) ---
sh = json.load(open(f"{ASMOUT}/runs_{TAG_H}/summary_B.json"))
sm = json.load(open(f"{ASMOUT}/runs_{TAG_M}/summary_B.json"))
losses_h = [sh["targets"][t]["loss"] for t in TARGETS]
losses_m = [sm["targets"][t]["loss"] for t in TARGETS]
print("=" * 70)
print("P1c 验收 v3 (5 目标 stress loss + 5Ro NMSE)")
print("=" * 70)
print()
print(f"  {'T':<4}  {'heur':>9}  {'minx':>9}  {'delta':>9}  yw_h yw_m  n_h n_m")
yh_y = []  # yw presence
ym_y = []
for i, t in enumerate(TARGETS):
    h = sh["targets"][t]
    m = sm["targets"][t]
    d = m["loss"] - h["loss"]
    s = "+" if d >= 0 else ""
    yh = "Y" if "yw" in h["dsl"] else "N"
    ym = "Y" if "yw" in m["dsl"] else "N"
    yh_y.append(yh); ym_y.append(ym)
    print(f"  {t:<4}  {h['loss']:>9.4f}  {m['loss']:>9.4f}  {s}{d:>8.4f}  {yh:>3} {ym:>3}  {h['complexity']['nodes']:>3} {m['complexity']['nodes']:>3}")
mh = np.mean(losses_h); mm = np.mean(losses_m)
print(f"  mean  {mh:>9.4f}  {mm:>9.4f}  {mm-mh:+9.4f}")

# --- 5 目标 5 Ro NMSE ---
print()
print("5 Ro NMSE 全 5 目标对比 (best['metrics']) — 总计 25 个 cell, 5 Ro 列:")
print(f"  {'T':<4}  {'后端':<10}  ", "  ".join(f"Ro{ro:<5g}" for ro in RO_CASES), "  mean")
def show_nmse(targets_summary, label):
    for t in TARGETS:
        ms = targets_summary[t]["metrics"]
        vals = [ms.get(f"Ro{ro:g}_nmse", np.nan) for ro in RO_CASES]
        print(f"  {t:<4}  {label:<10}  ", "  ".join(f"{v:.3f}".rjust(7) for v in vals),
              f"  {np.nanmean(vals):.3f}")
print("heuristic:")
show_nmse(sh["targets"], "heur")
print("minimax:")
show_nmse(sm["targets"], "minx")

# V2 vw 中心峰 (从重构 5 Ro T5g 公式驱动)
print()
print("V2: vw 中心峰 (|y/h|<0.15, T5g 公式驱动) — 5 Ro")
# 之前 v2 跑过的结果已存 — 重做: 用 theta + 编译 + 5 Ro featA → a_pred → b_hat
from lgf.compiler import CompiledGraph
from lgf.dsl import parse_dsl
import re
zh = np.load(f"{ASMOUT}/dataset_pope_gsonly.npz")
zg = np.load(f"{ASMOUT}/dataset_pope3_wy.npz")
print(f"  {'Ro':<5}  {'DNS':>5}  {'heur':>5}  {'minx':>5}  h_pass  m_pass")
ph, pm = 0, 0
for i, ro in enumerate(RO_CASES):
    p = f"c{i}_"
    featA = zh[p + "featA"]
    # 5 个基的 B_T
    bsel = []
    for n in TARGETS:
        if n == "T5g":
            bsel.append(zg[p + "B_T5g"])
        else:
            bsel.append(zh[p + f"B_{n}"])
    M = np.stack(bsel, axis=1)  # (N, 5, 6)
    # T5g 公式预测
    a_pred = np.zeros((len(featA), 5))
    for ti, (tag, ts) in enumerate([(TAG_H, sh), (TAG_M, sm)]):
        # 只需要 T5g 公式 (index 4), 但前 4 个基 a_m=0 简化 — 不严格, 但 vw 公式主导
        # 严格做: 5 目标都跑 — 简化先只跑 T5g (因为 vw = a_5g * T5g + 其他弱贡献)
        # 正确做法: 5 个目标都重构 a_m 再算 b_hat.  5 Ro × 5 target × 2 后端 = 50 次编译, 慢
        # 简化: 加载 T5g best 公式, 直接预测 a_5g 用来算 b_vw 的 T5g 贡献 (主导项)
        pass
    # 完整重构: 5 target × 2 后端
    a_h = np.zeros((len(featA), 5))
    a_m = np.zeros((len(featA), 5))
    for ti, t in enumerate(TARGETS):
        for tag, a_arr, ts in [(TAG_H, a_h, sh), (TAG_M, a_m, sm)]:
            best = ts["targets"][t]
            dsl = best["dsl"]
            vn = ts["var_names"]
            g = parse_dsl(dsl, var_names=vn)
            cg = CompiledGraph(g, backend="numpy")
            n_g = cg.n_g; n_c = cg.n_c
            tl = np.concatenate([best["theta"], np.zeros(n_c)])
            try:
                a_arr[:, ti] = cg.eval_case(tl, featA)
            except Exception:
                a_arr[:, ti] = 0.0
    b_h = np.einsum("nkm,nk->nm", M, a_h)
    b_m = np.einsum("nkm,nk->nm", M, a_m)
    b6 = zh[p + "b6"]
    yh = zh[p + "yh"]
    cn = np.abs(yh) < 0.15
    dns = b6[cn, 5].mean()
    vh = b_h[cn, 5].mean()
    vm = b_m[cn, 5].mean()
    hp = "Y" if vh > 0.03 else "N"
    mp = "Y" if vm > 0.03 else "N"
    if vh > 0.03: ph += 1
    if vm > 0.03: pm += 1
    print(f"  Ro{ro:<4g}  {dns:.3f}  {vh:.3f}  {vm:.3f}    {hp}       {mp}")
print(f"  vw 中心峰达标: heur {ph}/5, minx {pm}/5")

# V3 yw 占用
print()
print("V3: yw 占用 (best 公式含 yw)")
hy = sum(1 for t in TARGETS if "yw" in sh["targets"][t]["dsl"])
my = sum(1 for t in TARGETS if "yw" in sm["targets"][t]["dsl"])
print(f"  heuristic: {hy}/5, minimax: {my}/5")

# V4 反演健康度
print()
print("V4: 反演健康度 max|a|@fit")
amax = []
for i in range(5):
    p = f"c{i}_"
    vf = zh[p + "valid_fit"]
    if vf.sum() == 0: continue
    amax.append(float(np.nanmax(np.abs(zh[p + "a"][vf]))))
print(f"  max across 5 Ro: {max(amax):.3g}  (要求 < 1e3)")

# 结论
print()
print("=" * 70)
print("结论摘要")
print("=" * 70)
print(f"  5 目标 loss 均值:  heur={mh:.4f}  minx={mm:.4f}  d={mm-mh:+.4f} ({(mm-mh)/mh*100:+.1f}%)")
print(f"  vw 中心峰 5/5:    heur={ph}/5  minx={pm}/5")
print(f"  yw 占用 5/5:      heur={hy}/5  minx={my}/5")
