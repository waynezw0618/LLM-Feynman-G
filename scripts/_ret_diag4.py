# -*- coding: utf-8 -*-
"""_ret_diag4.py — 用真壁距 yw 重算 Re_t(Ro, yw) 剖面; 检查 d[:,0] 范围."""
import sys, os
import numpy as np
sys.path.insert(0, r"d:\AI\LLM-Feynman-G\scripts")
from rc_asm_prep import load_case, RO_CASES, DATA_DIR, FILES

d0 = np.loadtxt(os.path.join(DATA_DIR, FILES[30.0]), comments="%")
print(f"d[:,0] range: [{d0[:,0].min():.4f}, {d0[:,0].max():.4f}], N={d0.shape[0]}, "
      f"U range [{d0[:,1].min():.3f},{d0[:,1].max():.3f}]")

print(f"\n{'Ro':>6} " + " ".join(f"{'yw='+str(w):>9}" for w in (1, 2, 5, 10, 30, 60, 120, 180)))
for ro in RO_CASES:
    fc = load_case(ro, variant="vort")
    k, eps = fc["k"], fc["eps"]
    ret = k**2 / np.maximum(eps, 1e-30)
    yw = (1.0 - np.abs(fc["yh"])) * 180.0
    vals = []
    for w in (1, 2, 5, 10, 30, 60, 120, 180):
        vals.append(ret[np.argmin(np.abs(yw - w))])
    print(f"{ro:>6} " + " ".join(f"{v:>9.3g}" for v in vals))

# Re_t 平台值 (中心) 随 Ro 的标度
print("\nRo_tau vs Re_t(center):")
for ro in RO_CASES:
    fc = load_case(ro, variant="vort")
    ret = fc["k"]**2 / np.maximum(fc["eps"], 1e-30)
    yw = (1.0 - np.abs(fc["yh"])) * 180.0
    print(f"  Ro={ro:>5}: Re_t_ctr={ret[np.argmin(np.abs(yw-180))]:.1f}")
