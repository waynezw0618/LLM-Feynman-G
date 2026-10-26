"""_gs_compare.py — v/e/g 三种 W 取法的 5 基反演健康度对照.

子集: sel={T1,T2v,T3v,T4v,T5v} (vort, 当前选中)
      gs ={T1,T2g,T3g,T4g,T5g} (Gatski-Speziale, W*=W+1·Om)
      ext={T1,T2e,T3e,T4e,T5e} (黄/马 扩展内禀, W+2·Om, = 旧 ASM 5 基)
口径: 与 rc_pope_prep v2 一致 (min-norm lstsq rcond=1e-10,
      valid_fit = cond<=1e4 & max|a_m|<=300)。
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "asm")

SUBSETS = {
    "sel(vort)": ["T1", "T2v", "T3v", "T4v", "T5v"],
    "gs(W+1Om)": ["T1", "T2g", "T3g", "T4g", "T5g"],
    "ext(W+2Om)": ["T1", "T2e", "T3e", "T4e", "T5e"],
}
COND_MAX, AMAX_CAP = 1e4, 300.0

z = np.load(os.path.join(OUT, "dataset_pope3.npz"))
ros = [float(z[f"c{i}_ro"]) for i in range(5)]
print(f"{'subset':<12} {'rel_resid':>10} {'p90_cond':>9} {'max|a|@fit':>11} "
      f"{'nw_fit%':>8} {'a_m 相关(sel)':>14}")
for sname, names in SUBSETS.items():
    tot_sse = tot_ssb = 0.0
    all_c = []
    amax_all = 0.0
    nw_tot = nw_fit = 0
    a_corr = []
    for i in range(5):
        p = f"c{i}_"
        b6 = z[p + "b6"]
        M = np.stack([z[p + "B_" + n] for n in names], axis=2)  # (N,6,K)
        N = b6.shape[0]
        sse = ssb = 0.0
        a = np.full((N, len(names)), np.nan)
        pt_c = np.zeros(N)
        for j in range(N):
            s = np.linalg.svd(M[j], compute_uv=False)
            if s[-1] > 0:
                pt_c[j] = s[0] / s[-1]
            if int(np.sum(s / max(s[0], 1e-300) > 1e-10)) < len(names):
                continue
            a[j], *_ = np.linalg.lstsq(M[j], b6[j], rcond=1e-10)
            sse += float(np.sum((M[j] @ a[j] - b6[j]) ** 2))
        ssb = float(np.sum(b6[~np.isnan(a[:, 0])] ** 2))
        tot_sse += sse
        tot_ssb += ssb
        valid = np.isfinite(a).all(1)
        all_c.extend(pt_c[valid].tolist())
        fitm = valid & (pt_c <= COND_MAX) & (np.nanmax(np.abs(a), axis=1) <= AMAX_CAP)
        if fitm.any():
            amax_all = max(amax_all, float(np.nanmax(np.abs(a[fitm]))))
        nw = np.abs(z[p + "yh"]) > 0.9
        nw_tot += int((nw & valid).sum())
        nw_fit += int((nw & fitm).sum())
        # 与 sel 基的 a_m 逐列相关 (只对共享结构 T1 有意义, 这里报整体形状相似度)
        if sname != "sel(vort)":
            asel = np.load(os.path.join(OUT, "dataset_pope3.npz"))[p + "a"]
            ok = valid & np.isfinite(asel).all(1)
            if ok.any():
                r = [float(np.corrcoef(a[ok, m], asel[ok, m])[0, 1])
                     for m in range(len(names))]
                a_corr.append(np.nanmean(r))
    rel = float(np.sqrt(tot_sse / max(tot_ssb, 1e-300)))
    corr_s = f"{np.nanmean(a_corr):+.3f}" if a_corr else "  —"
    print(f"{sname:<12} {rel:>10.2e} {np.percentile(all_c, 90):>9.1f} "
          f"{amax_all:>11.3g} {100.0 * nw_fit / max(nw_tot, 1):>7.1f}% {corr_s:>14}")

# 每 Ro 的近壁 valid_fit 覆盖
print("\n每 Ro 近壁(|y/h|>0.9) valid_fit 覆盖率:")
print(f"{'Ro':<7}" + "".join(f"{sname:>14}" for sname in SUBSETS))
for i in range(5):
    p = f"c{i}_"
    row = f"{ros[i]:<7g}"
    for sname, names in SUBSETS.items():
        M = np.stack([z[p + "B_" + n] for n in names], axis=2)
        b6 = z[p + "b6"]
        N = b6.shape[0]
        a = np.full((N, len(names)), np.nan)
        pt_c = np.zeros(N)
        for j in range(N):
            s = np.linalg.svd(M[j], compute_uv=False)
            if s[-1] > 0:
                pt_c[j] = s[0] / s[-1]
            if int(np.sum(s / max(s[0], 1e-300) > 1e-10)) < len(names):
                continue
            a[j], *_ = np.linalg.lstsq(M[j], b6[j], rcond=1e-10)
        valid = np.isfinite(a).all(1)
        fitm = valid & (pt_c <= COND_MAX) & (np.nanmax(np.abs(a), axis=1) <= AMAX_CAP)
        nw = np.abs(z[p + "yh"]) > 0.9
        row += f"{100.0 * (nw & fitm).sum() / max((nw & valid).sum(), 1):>13.1f}%"
    print(row)
