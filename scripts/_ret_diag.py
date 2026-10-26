# -*- coding: utf-8 -*-
"""_ret_diag.py — Re_t 候选特征可行性 + cond_max 放松影响的快速诊断."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from rc_asm_prep import load_case, RO_CASES

print("=== 1) Re_t = k^2/(nu*eps), 粘性单位 nu=1, 各工况范围 ===")
print(f"{'Ro':>6} {'Re_t_min':>10} {'Re_t@yp=5':>10} {'Re_t@yp=30':>11} {'Re_t_ctr':>10} {'Re_t_max':>10}")
all_Ro_loc, all_ret, all_yp, all_yw, all_eta2, all_zeta2e = [], [], [], [], [], []
cases = []
for ro in RO_CASES:
    fc = load_case(ro, variant="vort")
    cases.append(fc)
    k, eps, yp = fc["k"], fc["eps"], fc["yp"]
    ret = k**2 / np.maximum(eps, 1e-30)          # nu=1 (wall units)
    om2 = ro / 180.0                              # 2*Omega (wall units): Ro_tau/h+ = ro/180
    ro_loc = om2 * k / np.maximum(eps, 1e-30)     # 局部旋转数 2*Omega*k/eps
    ctr = np.argmin(np.abs(fc["yh"]))
    i5 = np.argmin(np.abs(yp - 5)); i30 = np.argmin(np.abs(yp - 30))
    print(f"{ro:>6} {ret.min():>10.2f} {ret[i5]:>10.2f} {ret[i30]:>11.2f} "
          f"{ret[ctr]:>10.2f} {ret.max():>10.2f}")
    all_Ro_loc.append(ro_loc); all_ret.append(ret)
    all_yp.append(yp); all_yw.append(180.0 - np.abs(fc["yh"]) * 180.0)
    ke = (k / eps)[:, None, None]
    Ss = ke * fc["S"]; Wv = ke * fc["Ot"]
    all_eta2.append(np.einsum("nii->n", Ss @ Ss))
    all_zeta2e.append(-np.einsum("nii->n", Wv @ Wv))
    ro_loc_p = ro_loc / (1.0 + np.abs(ro_loc))    # 饱和映射示例

ret = np.concatenate(all_ret); yp = np.concatenate(all_yp); yw = np.concatenate(all_yw)
eta2 = np.concatenate(all_eta2); z2e = np.concatenate(all_zeta2e)
rol = np.concatenate(all_Ro_loc)
m = np.isfinite(ret) & (ret > 0)
def c(a, b):
    ma = m & np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ma], b[ma])[0, 1]
print("\n=== 2) Re_t 与现有特征相关性 (全部 5 工况拼接) ===")
print(f"corr(Re_t, yp)        = {c(ret, yp):+.3f}")
print(f"corr(Re_t, yw)        = {c(ret, yw):+.3f}")
print(f"corr(Re_t, eta2)      = {c(ret, eta2):+.3f}")
print(f"corr(Re_t, zeta2v)    = {c(ret, z2e):+.3f}")
print(f"corr(1/Re_t, eta2)    = {c(1.0/ret, eta2):+.3f}")
print(f"corr(1/Re_t, zeta2v)  = {c(1.0/ret, z2e):+.3f}")
print(f"\nRo_local = 2*Om*k/eps: range [{rol.min():.3f}, {rol.max():.3f}], "
      f"每工况中心值: " + ", ".join(f"{r[cases[i]['yh'].argmin()]:.3f}" for i, r in enumerate(all_Ro_loc)))

print("\n=== 3) cond_max 放松影响 (选中基底 T1,T2v,T3v,T4v,T5v, 反演 a_m) ===")
from rc_pope_prep import invert_case, pointwise_cond
SEL = ["T1", "T2v", "T3v", "T4v", "T5v"]
print(f"{'cond_max':>9} {'nw_fit%':>8} {'all_fit%':>9} {'max|a|@fit':>11} {'p99|a|':>9}")
for cap in (1e3, 1e4, 1e5, 1e6, np.inf):
    amax = 0.0; nw_tot = nw_fit = 0; tot = fit = 0; p99 = []
    for fc in cases:
        ke = (fc["k"] / fc["eps"])[:, None, None]
        Ss = ke * fc["S"]; Wv = ke * fc["Ot"]
        from rc_pope_prep import pope_tensors, sym6
        b6d = dict((n if n == "T1" else n + "v", sym6(T))
                   for n, T in pope_tensors(Ss, Wv))
        names = [n if n != "T1" else "T1" for n in SEL]
        a, _ = invert_case(fc["b6"], b6d, SEL)
        cnd = pointwise_cond(b6d, SEL)
        phys = (fc["k"] > 1e-4) & (fc["eps"] > 1e-12) & np.isfinite(fc["b6"]).all(1) \
               & np.isfinite(a).all(1)
        fitm = phys & (cnd <= cap) & (np.nanmax(np.abs(a), 1) <= 300.0)
        if fitm.any():
            amax = max(amax, float(np.nanmax(np.abs(a[fitm]))))
            p99.append(np.percentile(np.abs(a[fitm]), 99))
        nw = np.abs(fc["yh"]) > 0.9
        nw_tot += int((nw & phys).sum()); nw_fit += int((nw & fitm).sum())
        tot += int(phys.sum()); fit += int(fitm.sum())
    p99v = max(p99) if p99 else float("nan")
    print(f"{cap:>9g} {100*nw_fit/max(nw_tot,1):>7.1f}% {100*fit/max(tot,1):>8.1f}% "
          f"{amax:>11.3g} {p99v:>9.3g}")
