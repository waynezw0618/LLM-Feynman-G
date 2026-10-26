"""rc_pope_gsbuild.py — GS 组合实验数据集: T5v->T5g 换基 + ret/roloc 特征.

设计依据 (_gs_diag.py, 2026-10-04):
  槽道对称性下 5 基表达性等价 (所有 v/e/g 换基变体 rel_resid 均数值饱和
  7.7e-8), 选基纯由条件数/覆盖决定. GS (Gatski-Speziale W*=Wbar+1*eps*Om)
  单换基里 T5v->T5g 是唯一零近壁覆盖损失 (nw_fit 88.9% 不变, p90 340.8)
  的入口, 且 T5 = W*S^2 - S^2*W 正是 vw 支撑基 —— GS 修正把系统旋转显式
  注入其构造, 物理上对症 TG 涡 vw 应力.

入基判据 (等价于 rc_pope_prep.py v3 段2 --p90_slack 200 的放松判据):
  p90_cond 340.8 < 200 x base 81.6 (段1 的 0.7x 不放行, 段2 放松放行)
  且 coverage 守门: nw_fit% 无跌幅 (88.9 -> 88.9), max|a|@fit 187 <= 300.

特征: 11 不变量 (v/e/g 三系) + yw + ret + roloc = 14 列.
  (vs 基线 dataset_pope_wy 9 列: 8 不变量 + yw; vs 失败实验 wy_rt 11 列)

用法: python scripts/rc_pope_gsbuild.py
输出: outputs/asm/dataset_pope_gsrt.npz (+ prep_report_gsrt.json)
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_prep import load_case, RO_CASES

OUT_DIR = os.path.join(REPO, "outputs", "asm")
SRC = "dataset_pope3_wy.npz"
DST = "dataset_pope_gsrt.npz"
SEL = ["T1", "T2v", "T3v", "T4v", "T5g"]
FEATS_ADD = ["ret", "roloc"]
COND_MAX, AMAX_CAP = 1e4, 300.0


def invert(b6, basis6, names):
    """逐点 min-norm lstsq 反演 a_m (rank==K 的点有效). 复刻 prep 逻辑."""
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
    src = dict(np.load(os.path.join(OUT_DIR, SRC)))
    out = dict(src)
    out["sel_basis"] = np.array(SEL)

    kb = [str(x) for x in src["featB_keys"]]
    for f in FEATS_ADD:
        assert f not in kb, f"{f} 已存在"
    out["featB_keys"] = np.array(kb + FEATS_ADD)
    out["featA_keys"] = np.array(kb + FEATS_ADD)

    om_eff = {ro: load_case(ro, variant="vort")["Om_eff"] for ro in RO_CASES}

    report = {"src": SRC, "selected_basis": SEL,
              "basis_rationale": "T5v->T5g: GS 单换基唯一零 nw_fit 损失入口 "
                                 "(_gs_diag 2026-10-04: p90 340.8, nw_fit 88.9%, "
                                 "max|a| 187); T5=WS2-S2W 为 vw 支撑基, GS 修正 "
                                 "W*=Wbar+eps*Om 将系统旋转显式注入",
              "p90_relax_check": "p90 340.8 < 200x81.6 (段2 放松判据通过; "
                                 "段1 0.7x 判据不放行 → 本换基即 p90 放松的直接产物)",
              "features": kb + FEATS_ADD}
    print(f"[gsbuild] 基底: {SEL}")
    print(f"[gsbuild] 特征: {kb + FEATS_ADD} ({len(kb)+len(FEATS_ADD)} 列)")

    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        ro = float(src[p + "ro"])
        b6 = src[p + "b6"]
        basis6 = {n: src[p + "B_" + n] for n in SEL}
        a, rankv = invert(b6, basis6, SEL)
        valid = rankv & (src[p + "k"] > 1e-4) & (src[p + "eps"] > 1e-12) \
            & np.isfinite(b6).all(1) & np.isfinite(a).all(1)
        M = np.stack([basis6[n] for n in SEL], axis=2)
        cnd = np.array([np.linalg.svd(M[j], compute_uv=False)[0]
                        / max(np.linalg.svd(M[j], compute_uv=False)[-1], 1e-300)
                        for j in range(len(b6))])
        amax_row = np.nanmax(np.abs(a), axis=1)
        valid_fit = valid & (cnd <= COND_MAX) & (amax_row <= AMAX_CAP)
        # 反演相对残差
        invres = np.full(valid.shape, np.nan)
        if valid.any():
            Mv = M[valid]
            bhat = np.einsum("nmk,nk->nm", Mv, a[valid])
            invres[valid] = (np.linalg.norm(bhat - b6[valid], axis=1)
                             / np.maximum(np.linalg.norm(b6[valid], axis=1), 1e-12))
        # 特征列
        k = src[p + "k"]
        eps = np.maximum(src[p + "eps"], 1e-30)
        ret = (k ** 2 / eps).astype(src[p + "featB"].dtype)
        roloc = (2.0 * om_eff[RO_CASES[i]] * k / eps).astype(src[p + "featB"].dtype)
        cols = np.stack([ret, roloc], axis=1)
        out[p + "featB"] = np.concatenate([src[p + "featB"], cols], axis=1)
        out[p + "featA"] = np.concatenate([src[p + "featA"], cols], axis=1)
        # 更新反演/掩码/基列
        out[p + "a"] = a
        out[p + "valid"] = valid
        out[p + "valid_fit"] = valid_fit
        out[p + "cond"] = cnd
        out[p + "invres"] = invres
        # 删除旧基的 T_/W_ 列, 写新基
        for n in ("T1", "T2v", "T3v", "T4v", "T5v"):
            out.pop(p + "T_" + n, None)
            out.pop(p + "W_" + n, None)
        for n in SEL:
            out[p + "T_" + n] = basis6[n]
            out[p + "W_" + n] = np.sum(basis6[n] ** 2, axis=1)
        nw = np.abs(src[p + "yh"]) > 0.9
        print(f"[gsbuild] Ro={ro:>5g}: valid {int(valid.sum())}/{len(valid)}, "
              f"valid_fit {int(valid_fit.sum())}, "
              f"近壁 {int((nw & valid).sum())}->{int((nw & valid_fit).sum())}, "
              f"max|a|@fit={np.nanmax(np.abs(a[valid_fit])):.3g}, "
              f"反演残差 med={np.nanmedian(invres):.2e}, "
              f"p90_cond={np.nanpercentile(cnd[valid], 90):.1f}")

    assert out["c0_featB"].shape[1] == len(out["featB_keys"])
    np.savez_compressed(os.path.join(OUT_DIR, DST), **out)
    with open(os.path.join(OUT_DIR, "prep_report_gsrt.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[gsbuild] -> {os.path.join(OUT_DIR, DST)}")


if __name__ == "__main__":
    main()
