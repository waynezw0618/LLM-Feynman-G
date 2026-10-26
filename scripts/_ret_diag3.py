# -*- coding: utf-8 -*-
"""_ret_diag3.py — 核对 k/eps 前几个点的原始值, 确认 Re_t 近壁形态不是列解析错误."""
import numpy as np
import os
DATA = r"d:\AI\LLM-Feynman-G\data"
for f in ("Retau180Ro7.5.txt", "Retau180Ro150.txt"):
    d = np.loadtxt(os.path.join(DATA, f), comments="%")
    yp = d[:, 0] * 180.0
    k = d[:, 9]
    print(f"\n== {f} (cols={d.shape[1]}) 前 8 点 ==")
    print(f"{'y/h':>8} {'yp':>7} {'k':>10} {'eps_raw?':>10} {'k^2/eps':>10}")
    # eps 由 rc_asm_prep 逻辑: -0.5*(diss_uu+vv+ww), 已在 load_case 内
    import sys
    sys.path.insert(0, r"d:\AI\LLM-Feynman-G\scripts")
    from rc_asm_prep import load_case, _bud, COMPS, TERMS
    fc = load_case(7.5 if "7.5" in f else 150.0, variant="vort")
    eps = fc["eps"]; kk = fc["k"]
    n = 8
    for i in range(n):
        print(f"{d[i,0]:>8.4f} {yp[i]:>7.2f} {kk[i]:>10.4g} {eps[i]:>10.4g} "
              f"{kk[i]**2/max(eps[i],1e-30):>10.4g}")
    # dissip 列原始值 (uu+vv+ww)
    dsum = sum(_bud(d, c, "dissip") for c in ("uu", "vv", "ww"))
    print("dissip_sum 前 8 点:", np.round(dsum[:8], 5))
