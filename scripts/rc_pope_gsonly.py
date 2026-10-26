"""rc_pope_gsonly.py — 归因实验: GS 换基单独 (T5v->T5g), 特征集 = 基线 9 列.

因子设计 (2026-10-06, gsrt 实验后归因):
  heuristic_pope_wy   : T5v, 9 列 (8 v-不变量 + yw)   <- P1a 基线
  heuristic_gsonly    : T5g, 9 列                      <- 本脚本: 隔离 GS 换基效应
  heuristic_pope_wy_rt: T5v, 11 列 (+ret/roloc)        <- 已有, 隔离特征效应
  heuristic_pope_gsrt : T5g, 14 列 (11 inv + yw + rt)  <- 已有, 组合
V2 vw 中心峰 4/5 的改善来自 GS 换基还是 ret/roloc, 由本脚本裁决.

SRC = dataset_pope_wy.npz (基线 9 列); B_T5g 取自 dataset_pope3_wy.npz.
用法: python scripts/rc_pope_gsonly.py
输出: outputs/asm/dataset_pope_gsonly.npz (+ prep_report_gsonly.json)
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

OUT_DIR = os.path.join(REPO, "outputs", "asm")
SRC = "dataset_pope_wy.npz"
SRC_G = "dataset_pope3_wy.npz"          # T5g 张量来源
DST = "dataset_pope_gsonly.npz"
SEL = ["T1", "T2v", "T3v", "T4v", "T5g"]
COND_MAX, AMAX_CAP = 1e4, 300.0


def invert(b6, basis6, names):
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
    src_g = np.load(os.path.join(OUT_DIR, SRC_G))
    out = dict(src)
    out["sel_basis"] = np.array(SEL)

    kb = [str(x) for x in src["featB_keys"]]
    out["featB_keys"] = np.array(kb)
    out["featA_keys"] = np.array(kb)

    report = {"src": SRC, "selected_basis": SEL,
              "basis_rationale": "归因实验: T5v->T5g GS 单换基, 特征集与基线"
                                 " dataset_pope_wy 严格一致 (9 列), 隔离换基效应",
              "features": kb}
    print(f"[gsonly] 基底: {SEL}, 特征: {kb}")

    for i in range(5):
        p = f"c{i}_"
        b6 = src[p + "b6"]
        basis6 = {n: src[p + "B_" + n] for n in SEL if (p + "B_" + n) in src}
        if "T5g" not in basis6:
            basis6["T5g"] = src_g[p + "B_T5g"]
        a, rankv = invert(b6, basis6, SEL)
        valid = rankv & (src[p + "k"] > 1e-4) & (src[p + "eps"] > 1e-12) \
            & np.isfinite(b6).all(1) & np.isfinite(a).all(1)
        M = np.stack([basis6[n] for n in SEL], axis=2)
        cnd = np.array([np.linalg.svd(M[j], compute_uv=False)[0]
                        / max(np.linalg.svd(M[j], compute_uv=False)[-1], 1e-300)
                        for j in range(len(b6))])
        amax_row = np.nanmax(np.abs(a), axis=1)
        valid_fit = valid & (cnd <= COND_MAX) & (amax_row <= AMAX_CAP)
        out[p + "a"] = a
        out[p + "valid"] = valid
        out[p + "valid_fit"] = valid_fit
        out[p + "cond"] = cnd
        for n in ("T1", "T2v", "T3v", "T4v", "T5v"):
            out.pop(p + "T_" + n, None)
            out.pop(p + "W_" + n, None)
        for n in SEL:
            out[p + "T_" + n] = basis6[n]
            out[p + "W_" + n] = np.sum(basis6[n] ** 2, axis=1)
            # P1c 数据完整性修复: B_ 基底列也写盘 (此前漏写 B_T5g, 外部验收
            # 只能从 dataset_pope3_wy.npz 兜底; B_T1..B_T5v 从 src 继承故无碍)
            out[p + "B_" + n] = basis6[n]
        nw = np.abs(src[p + "yh"]) > 0.9
        print(f"[gsonly] c{i}: valid_fit {int(valid_fit.sum())}, "
              f"近壁 {int((nw & valid).sum())}->{int((nw & valid_fit).sum())}, "
              f"max|a|@fit={np.nanmax(np.abs(a[valid_fit])):.3g}, "
              f"p90_cond={np.nanpercentile(cnd[valid], 90):.1f}")

    assert out["c0_featB"].shape[1] == len(out["featB_keys"])
    np.savez_compressed(os.path.join(OUT_DIR, DST), **out)
    with open(os.path.join(OUT_DIR, "prep_report_gsonly.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[gsonly] -> {os.path.join(OUT_DIR, DST)}")


if __name__ == "__main__":
    main()
