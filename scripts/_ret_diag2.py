# -*- coding: utf-8 -*-
"""_ret_diag2.py — 修正: Ro_local 中心值取 y/h=0.5; 近壁 Re_t 过渡区宽度."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from rc_asm_prep import load_case, RO_CASES

print(f"{'Ro':>6} {'Ro_loc@ctr':>10} {'Ro_loc@yp30':>11} {'Ro_loc_max':>10} "
      f"{'Re_t@yp1':>9} {'Re_t@yp2':>9} {'Re_t@yp5':>9} {'Re_t@yp10':>10}")
for ro in RO_CASES:
    fc = load_case(ro, variant="vort")
    k, eps, yp, yh = fc["k"], fc["eps"], fc["yp"], fc["yh"]
    ret = k**2 / np.maximum(eps, 1e-30)
    ro_loc = (ro / 180.0) * k / np.maximum(eps, 1e-30)
    ctr = np.argmin(np.abs(yh - 0.5))
    def at(y): return ret[np.argmin(np.abs(yp - y))]
    def at_r(y): return ro_loc[np.argmin(np.abs(yp - y))]
    print(f"{ro:>6} {ro_loc[ctr]:>10.3f} {at_r(30):>11.3f} {ro_loc.max():>10.3f} "
          f"{at(1):>9.2f} {at(2):>9.2f} {at(5):>9.2f} {at(10):>10.2f}")
