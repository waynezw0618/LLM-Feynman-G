# -*- coding: utf-8 -*-
"""_gs_diag.py — GS (Gatski-Speziale, W* = Wbar + 1*eps*Om) 张量进基可行性诊断.

问题: 28 候选池已含 T*g 系列, 但 p90 放松选基 (slack=200) 选中的是 T8e (ext),
没有任何 GS 张量入基. 本脚本量化: 基线 5 基的 GS 单换/双换变体的
rel_resid / p90_cond / nw_fit% / max|a|@fit, 判定 GS 张量是否有可行入口.

对照: baseline {T1,T2v,T3v,T4v,T5v} 与 p90relax {T1,T8e,T4v,T3v,T5v}.
"""
from __future__ import annotations
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_prep import load_case, sym6, RO_CASES
from rc_pope_prep import (pope_tensors, eval_subset, invert_case,
                          pointwise_cond, fit_coverage)

OUT_DIR = os.path.join(REPO, "outputs", "asm")
COND_MAX, AMAX_CAP = 1e4, 300.0

BASE = ["T1", "T2v", "T3v", "T4v", "T5v"]
VARIANTS = [
    ("baseline", ["T1", "T2v", "T3v", "T4v", "T5v"]),
    ("p90relax", ["T1", "T8e", "T4v", "T3v", "T5v"]),
    ("T2v->T2g", ["T1", "T2g", "T3v", "T4v", "T5v"]),
    ("T3v->T3g", ["T1", "T2v", "T3g", "T4v", "T5v"]),
    ("T4v->T4g", ["T1", "T2v", "T3v", "T4g", "T5v"]),
    ("T5v->T5g", ["T1", "T2v", "T3v", "T4v", "T5g"]),
    ("T2+T5->g", ["T1", "T2g", "T3v", "T4v", "T5g"]),
    ("T4+T5->g", ["T1", "T2v", "T3v", "T4g", "T5g"]),
    ("all->g",   ["T1", "T2g", "T3g", "T4g", "T5g"]),
    ("T5v->T5e", ["T1", "T2v", "T3v", "T4v", "T5e"]),
]


def main():
    cases = []
    for ro in RO_CASES:
        fc = load_case(ro, variant="vort")
        ke = (fc["k"] / fc["eps"])[:, None, None]
        Ss = ke * fc["S"]
        Wv = ke * fc["Ot"]
        We = fc["Ot"].copy()
        We[:, 1, 2] += 2.0 * fc["Om_eff"]
        We[:, 2, 1] -= 2.0 * fc["Om_eff"]
        We = ke * We
        Wg = fc["Ot"].copy()
        Wg[:, 1, 2] += 1.0 * fc["Om_eff"]
        Wg[:, 2, 1] -= 1.0 * fc["Om_eff"]
        Wg = ke * Wg
        cand = []
        for n, T in pope_tensors(Ss, Wv):
            cand.append((n + "v", sym6(T)))
        for n, T in pope_tensors(Ss, We):
            if n == "T1":
                continue
            cand.append((n + "e", sym6(T)))
        for n, T in pope_tensors(Ss, Wg):
            if n == "T1":
                continue
            cand.append((n + "g", sym6(T)))
        cand = [("T1" if n == "T1v" else n, V) for n, V in cand]
        fc["_b6"] = dict(cand)
        cases.append(fc)

    print(f"{'variant':<11} {'rel_resid':>10} {'p90_cond':>10} "
          f"{'nw_fit%':>8} {'all_fit%':>8} {'max|a|@fit':>11}")
    print("-" * 62)
    rows = {}
    for vname, vs in VARIANTS:
        r = eval_subset(cases, vs)
        cov = fit_coverage(cases, vs, COND_MAX, AMAX_CAP)
        rows[vname] = dict(basis=vs, rel_resid=r["rel_resid"],
                           p90_cond=r["p90_cond"], **cov)
        print(f"{vname:<11} {r['rel_resid']:>10.2e} {r['p90_cond']:>10.1f} "
              f"{cov['nw_fit_pct']:>7.1f}% {cov['all_fit_pct']:>7.1f}% "
              f"{cov['amax_fit']:>11.3g}")

    # GS 张量六分量非零结构 (Ro=30, 中心点) — 判断 vw 支撑
    fc = cases[2]  # Ro=30
    i0 = int(np.argmin(np.abs(fc["yh"])))
    print(f"\n[GS 结构] Ro=30, y/h={fc['yh'][i0]:+.3f} (中心) 各候选 vw 分量幅值:")
    for n in ("T2v", "T2g", "T2e", "T5v", "T5g", "T5e", "T8e", "T8g"):
        v6 = fc["_b6"][n][i0]
        print(f"  {n}: vw={v6[5]:+.4f}  uu={v6[0]:+.4f}  uv={v6[3]:+.4f}")

    import json
    with open(os.path.join(OUT_DIR, "_gs_diag.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\n-> {os.path.join(OUT_DIR, '_gs_diag.json')}")


if __name__ == "__main__":
    main()
