"""rc_asm_diag_B.py — 诊断方案B失败原因 (Q1) + Ω̃ vs 涡量 W 基底对比 (Q2).

诊断项:
  D1 公式奇异性: 检查 5 个 a_m 符号式在 η²,ζ²→0 (壁面/中心) 处是否发散 (1/η², ln ζ²)
  D2 误差放大: b = Σ a_m T_m, Δb_m = Δa_m·T_m — 哪个 a_m、哪个 y 区间贡献了主要误差
  D3 结构不可辨识: a_m 是否真是不变量的单值函数? (kNN 不变量近邻的 a_m 散布)
  D4 不变量退化: 2D 平均流下 5 个不变量实际独立维数 (互拟合 R²)

用法: python scripts/rc_asm_diag_B.py [--vort]   # --vort 时用 dataset_vort.npz
输出: outputs/asm/diag_B_report.md (+ stdout 摘要)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lgf import dsl as lgfdsl
from lgf.compiler import CompiledGraph

OUT = os.path.join(REPO, "outputs", "asm")
RO = [7.5, 15.0, 30.0, 75.0, 150.0]
SEL = ["S", "SOm", "S2", "Om2", "Om2S"]


def decode(dsl, var_names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return CompiledGraph(lgfdsl.parse_dsl(dsl, var_names))
        except Exception:
            return None


def eval_one(f, X, theta, ci):
    n_g, n_c = f.n_g, f.n_c
    th_g = theta[:n_g]
    th_c = theta[n_g + ci * n_c: n_g + (ci + 1) * n_c]
    try:
        return np.asarray(f.eval_case(np.concatenate([th_g, th_c]), X), dtype=float)
    except Exception:
        return None


def load_all(dsname, summ_dir):
    ds = np.load(os.path.join(OUT, dsname))
    summ = json.load(open(os.path.join(OUT, summ_dir, "summary_B.json"), encoding="utf-8"))
    return ds, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.npz")
    ap.add_argument("--tag", default="heuristic")
    ap.add_argument("--out", default="diag_B_report.md")
    args = ap.parse_args()

    ds, summ = load_all(args.dataset, f"runs_{args.tag}")
    featB_keys = [str(x) for x in ds["featB_keys"]]
    lines = [f"# 方案B 诊断报告 (dataset={args.dataset}, tag={args.tag})", ""]

    # ---------- D1: 公式奇异性 / a_m 预测范围 ----------
    lines += ["## D1. a_m 公式在评估点的奇异性与幅值", ""]
    lines += ["| a_m | 公式 | 真值范围 (5工况) | 预测范围 | 预测/真值 幅值比 |", "|---|---|---|---|---|"]
    preds = {}   # (ci, m) -> ypa
    truth_rng = {m: [] for m in SEL}
    pred_rng = {m: [] for m in SEL}
    for tgt, tinfo in summ["targets"].items():
        f = decode(tinfo["dsl"], featB_keys)
        theta = np.asarray(tinfo["theta"], dtype=float)
        mi = SEL.index(tgt)
        for ci in range(len(RO)):
            p = f"c{ci}_"; v = ds[p + "valid"]
            y_a = ds[p + "a"][:, mi][v]
            yp = eval_one(f, ds[p + "featB"][v], theta, ci) if f else None
            truth_rng[tgt].append((float(np.nanmin(y_a)), float(np.nanmax(y_a))))
            if yp is None:
                pred_rng[tgt].append((np.nan, np.nan))
            else:
                pred_rng[tgt].append((float(np.nanmin(yp)), float(np.nanmax(yp))))
                preds[(ci, mi)] = yp
        t_lo = min(a for a, b in truth_rng[tgt]); t_hi = max(b for a, b in truth_rng[tgt])
        p_lo = min((a for a, b in pred_rng[tgt] if np.isfinite(a)), default=np.nan)
        p_hi = max((b for a, b in pred_rng[tgt] if np.isfinite(b)), default=np.nan)
        ratio = (max(abs(p_lo), abs(p_hi)) / max(abs(t_lo), abs(t_hi), 1e-30)) \
            if np.isfinite(p_lo) else float("inf")
        dsl = (tinfo["dsl"] or "")[:70]
        lines.append(f"| {tgt} | `{dsl}` | [{t_lo:.3g}, {t_hi:.3g}] | "
                     f"[{p_lo:.3g}, {p_hi:.3g}] | {ratio:.3g} |")

    # 不变量取值范围 (奇异性来源定位)
    lines += ["", "不变量在 valid 点的范围 (检查 η²,ζ²→0 的奇异性触发):", ""]
    lines += ["| 不变量 | min | median | max |", "|---|---|---|---|"]
    for j, key in enumerate(featB_keys):
        vals = np.concatenate([ds[f"c{ci}_featB"][ds[f"c{ci}_valid"]][:, j] for ci in range(5)])
        lines.append(f"| {key} | {vals.min():.3g} | {np.median(vals):.3g} | {vals.max():.3g} |")

    # ---------- D2: 误差分解 ----------
    lines += ["", "## D2. b 重建误差按 a_m 与 y 区间分解", ""]
    lines += ["每个 cell = |Δa_m·T_m| 的中位数 (对 6 分量、5 工况、全部 valid 点), "
              "并给出最大误差出现的 y/h。", ""]
    lines += ["| a_m | med |Δa_m T_m| | max |Δa_m T_m| | max 出现在 y/h | 该点 Ro |", "|---|---|---|---|---|"]
    err_by_m = {m: [] for m in SEL}
    for ci in range(len(RO)):
        p = f"c{ci}_"; v = ds[p + "valid"]
        Tb = np.stack([ds[p + "T_" + m][v] for m in SEL], axis=1)   # (Nv,5,6)
        yh = ds[p + "yh"][v]
        for mi, m in enumerate(SEL):
            yp = preds.get((ci, mi))
            if yp is None or not np.all(np.isfinite(yp)):
                continue
            da = yp - ds[p + "a"][:, mi][v]                          # (Nv,)
            db = np.abs(da[:, None, None] * Tb[:, mi, :])            # (Nv,6)
            err_by_m[m].append((np.median(db), db.max(), yh[np.unravel_index(db.argmax(), db.shape)[0]],
                                RO[ci]))
    for m in SEL:
        if err_by_m[m]:
            med = np.median([e[0] for e in err_by_m[m]])
            mx = max(e[1] for e in err_by_m[m])
            i = int(np.argmax([e[1] for e in err_by_m[m]]))
            lines.append(f"| {m} | {med:.3g} | {mx:.3g} | {err_by_m[m][i][2]:.3g} | "
                         f"{err_by_m[m][i][3]:g} |")
    # 放大倍数: ||T||/||b||
    lines += ["", "误差放大背景: 基底张量范数 vs b 本身范数 (中位数, 5 工况合并):", ""]
    nT, nb = [], []
    for ci in range(len(RO)):
        p = f"c{ci}_"; v = ds[p + "valid"]
        Tb = np.stack([ds[p + "T_" + m][v] for m in SEL], axis=1)
        nT.append(np.median(np.linalg.norm(Tb, axis=2)))             # Frobenius per point
        nb.append(np.median(np.linalg.norm(ds[p + "b6"][v], axis=1)))
    lines.append(f"- med ||T^(m)|| (逐点 Frobenius) = {np.median(nT):.3g}, "
                 f"med ||b|| = {np.median(nb):.3g}, "
                 f"放大比 ≈ {np.median(nT)/np.median(nb):.3g}×"

                 )

    # ---------- D3: 结构不可辨识 (kNN) ----------
    lines += ["", "## D3. a_m 是否为不变量的单值函数? (kNN 歧义度检验)", "",
              "对每个工况: 在归一化 5 维不变量空间找 8 近邻, 比较近邻间 a_m 散布"
              "与全体 a_m 标准差。比值≈1 → 无歧义(局部可辨识); 比值远小于 1 → "
              "近邻 a_m 差异大 → 存在同不变量不同 a_m 的'多值性'。", ""]
    lines += ["| Ro | a_m | 邻内σ/全局σ | 同点位歧义对数 |", "|---|---|---|---|"]
    amb_scores = {m: [] for m in SEL}
    for ci in range(len(RO)):
        p = f"c{ci}_"; v = ds[p + "valid"]
        X = ds[p + "featB"][v].copy()
        # 归一化
        mu, sd = X.mean(0), X.std(0) + 1e-30
        Xn = (X - mu) / sd
        A = ds[p + "a"][v]                                          # (Nv,5)
        N = Xn.shape[0]
        # 全配对距离 (N~200, 可承受)
        D = np.sqrt(((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1)[:, :8]                            # 8 近邻
        for mi, m in enumerate(SEL):
            gsig = float(np.std(A[:, mi])) + 1e-30
            loc = []
            for i in range(N):
                loc.append(np.std(A[nn[i], mi]))
            loc = np.array(loc)
            amb_scores[m].append(float(np.median(loc) / gsig))
            # 歧义对: 距离<0.05 但 a_m 差 > 1 个全局σ
            ii, jj = np.where((D < 0.05))
            keep = ii < jj
            ii, jj = ii[keep], jj[keep]
            namb = int(np.sum(np.abs(A[ii, mi] - A[jj, mi]) > gsig)) if ii.size else 0
            lines.append(f"| {RO[ci]:g} | {m} | {np.median(loc)/gsig:.2f} | "
                         f"{namb}/{ii.size} |")
    lines += ["", "解读: '邻内σ/全局σ' 接近 1 说明近邻 a_m 依旧散开 → "
              "局部不变量不足以决定 a_m (历史/壁面记忆效应)。", ""]

    # ---------- D4: 不变量独立性 ----------
    lines += ["## D4. 5 个不变量的有效独立维数 (互拟合 R², 5 工况合并)", ""]
    Xall = np.concatenate([ds[f"c{ci}_featB"][ds[f"c{ci}_valid"]] for ci in range(5)])
    keys = featB_keys
    lines += ["| 目标 | 用其余 4 个线性拟合 R² |", "|---|---|"]
    for j, k in enumerate(keys):
        others = [i for i in range(len(keys)) if i != j]
        Amat = np.column_stack([Xall[:, others], np.ones(len(Xall))])
        coef, *_ = np.linalg.lstsq(Amat, Xall[:, j], rcond=None)
        r = 1 - np.var(Xall[:, j] - Amat @ coef) / (np.var(Xall[:, j]) + 1e-30)
        lines.append(f"| {k} | {r:.4f} |")

    rep = os.path.join(OUT, args.out)
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print("\n".join(lines))
    print(f"\n[diag] -> {rep}")


if __name__ == "__main__":
    main()
