"""P1d-V3a 前置复现: 为何 heuristic 臂 M_GUIDE 也零追加 (n_pop 恒 40)?

复刻 rc_asm_run.py --scheme B --fitness stress --fit_mask valid_fit
  --wall_boost 8 --w_vw 5 --w_uw 3 (dataset_pope_gsonly.npz, target T5g/T1)
然后按 pipeline_lgf.py:136-154 的顺序执行 analyze -> M_GUIDE -> apply_mutations,
任何一步抛异常都会被 run_lgf 的 except: pass 吞掉, 这里全部打出来.
"""
import json
import os
import sys
import traceback

import numpy as np

REPO = r"d:\AI\LLM-Feynman-G"
sys.path.insert(0, REPO)
ASMOUT = os.path.join(REPO, "outputs", "asm")

from lgf import llm_ops, attribution
from lgf.dsl import parse_dsl

RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
COMP6 = ["uu", "vv", "ww", "uv", "uw", "vw"]
COMP_W = {c: (5.0 if c == "vw" else 3.0 if c == "uw" else 1.0) for c in COMP6}


def build_cases(ds, ycol, sel_names):
    cs = []
    for i in range(len(RO_CASES)):
        p = f"c{i}_"
        v = ds[p + "valid_fit"]
        X = ds[p + "featB"][v]
        col = sel_names.index(ycol)
        y = ds[p + "a"][:, col][v]
        c = dict(X=X, y=y, name=f"Ro{RO_CASES[i]:g}")
        yh = np.abs(ds[p + "yh"][v])
        wall_mult = 1.0 + (8.0 - 1.0) * (yh > 0.7).astype(float)
        Tm = ds[p + "T_" + ycol][v]
        w = np.zeros(len(y))
        for kk, cc in enumerate(COMP6):
            w += COMP_W[cc] * Tm[:, kk] ** 2
        c["w"] = w * wall_mult
        cs.append(c)
    return cs


ds = np.load(os.path.join(ASMOUT, "dataset_pope_gsonly.npz"))
sel_names = [str(x) for x in ds["sel_basis"]]
var_names = [str(x) for x in ds["featB_keys"]]
client = llm_ops.make_client("heuristic")

for tgt, run_dir in [("T5g", "runs_heuristic_gsonly_v2"), ("T1", "runs_heuristic_gsonly_v2")]:
    print("=" * 70)
    print(f"target={tgt}")
    fin = json.load(open(os.path.join(ASMOUT, run_dir, f"B_a_{tgt}_final.json"),
                         encoding="utf-8"))
    best = fin["best"]
    g = parse_dsl(best["dsl"], var_names)
    g.theta = np.asarray(best["theta"], dtype=float)
    cs = build_cases(ds, tgt, sel_names)
    rng = np.random.default_rng(0)

    print("[1] attribution.analyze ...")
    try:
        rep = attribution.analyze(g, g.theta, cs, var_names)
        print(f"    OK keys={sorted(rep.keys())}")
    except Exception:
        print("    *** EXCEPTION (这就是 pipeline_lgf.py:155 except:pass 吞掉的) ***")
        traceback.print_exc()
        continue

    rep["forbidden_ops"] = ["sub", "div"] if tgt == "T5g" else []
    print("[2] M_GUIDE (heuristic client) ...")
    try:
        mutations = llm_ops.M_GUIDE(client, [(g, best["loss"])], rep)
        print(f"    OK n_mutations={len(mutations)}: {mutations}")
    except Exception:
        print("    *** EXCEPTION ***")
        traceback.print_exc()
        continue

    print("[3] apply_mutations top-1 ...")
    try:
        ng = llm_ops.apply_mutations(g, mutations, rng,
                                     forbidden_ops=rep["forbidden_ops"])
        same = ng is not None and ng.to_dsl() == g.to_dsl()
        print(f"    OK ng={'None' if ng is None else 'clone-equal' if same else 'changed'}"
              f" edges={None if ng is None else ng.count_edges()}")
    except Exception:
        print("    *** EXCEPTION ***")
        traceback.print_exc()
print("done")
