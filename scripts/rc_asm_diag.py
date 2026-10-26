"""rc_asm_diag.py — 诊断方案A预测为何不在图上。"""
from __future__ import annotations
import json, os, sys, warnings
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from lgf import dsl as lgfdsl
from lgf.compiler import CompiledGraph

OUT = os.path.join(REPO, "outputs", "asm")
ds = np.load(os.path.join(OUT, "dataset.npz"))
featA_keys = [str(x) for x in ds["featA_keys"]]
RO = [7.5, 15.0, 30.0, 75.0, 150.0]
SIX = ["uu", "vv", "ww", "uv", "uw", "vw"]
run_dir = os.path.join(OUT, "runs_heuristic")

print(f"{'target':<6} {'Ro':>6} {'yDNS_range':<24} {'yp_range':<24} {'finite':>6}")
for tgt in SIX:
    p_final = os.path.join(run_dir, f"A_b_{tgt}_final.json")
    if not os.path.exists(p_final):
        print(f"{tgt}  MISSING"); continue
    s = json.load(open(p_final, encoding="utf-8"))
    b = s["best"]
    dsl = b["dsl"]; theta = np.asarray(b["theta"], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            g = lgfdsl.parse_dsl(dsl, featA_keys); f = CompiledGraph(g)
        except Exception as e:
            print(f"{tgt}  DSL FAIL: {e}"); continue
    L = theta.size; nc = 4
    for ci in range(len(RO)):
        p_ds = f"c{ci}_"
        v = ds[p_ds + "valid"]; X = ds[p_ds + "featA"][v]
        y_b = ds[p_ds + "b6"][:, SIX.index(tgt)][v]
        tl = np.concatenate([theta[:L-nc*len(RO)], theta[L-nc*len(RO)+ci*nc : L-nc*len(RO)+(ci+1)*nc]])
        try:
            yp = np.asarray(f.eval_case(tl, X), dtype=float)
        except Exception as e:
            print(f"{tgt} Ro{RO[ci]:g}  eval FAIL: {e}"); continue
        finite = int(np.sum(np.isfinite(yp)))
        yp_f = yp[np.isfinite(yp)]
        yr = (float(np.min(y_b)), float(np.max(y_b)))
        ypr = (None if yp_f.size == 0 else (float(np.min(yp_f)), float(np.max(yp_f))))
        print(f"{tgt:<6} Ro{RO[ci]:g} y[{yr[0]:.3g},{yr[1]:.3g}] "
              f"yp[{'-' if ypr[0] is None else f'{ypr[0]:.3g}'},"
              f"{'-' if ypr[1] is None else f'{ypr[1]:.3g}'}] "
              f"finite={finite}/{yp.size}")