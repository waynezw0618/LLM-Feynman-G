# 前置诊断: 逐点 cond 分布 (选定5基), a_m 量级, 近壁点保留率
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from rc_asm_prep import load_case, sym6, RO_CASES
from rc_pope_prep import pope_tensors, invert_case

SEL = ["T1", "T2v", "T3v", "T4v", "T5v"]

print(f"{'Ro':>6} {'N':>4} {'c>1e3':>6} {'c>1e4':>6} {'c>1e6':>6} "
      f"{'nw_all':>7} {'nw_c<1e4':>9} {'nw_c<1e3':>9} {'max|a|(c<1e4)':>15} {'max|a|(c<1e3)':>15}")
for ro in RO_CASES:
    fc = load_case(ro, variant="vort")
    ke = (fc["k"] / fc["eps"])[:, None, None]
    Ss = ke * fc["S"]
    Wv = ke * fc["Ot"]
    b6 = {}
    for n, T in pope_tensors(Ss, Wv):
        b6[("T1" if n == "T1" else n + "v")] = sym6(T)
    M = np.stack([b6[n] for n in SEL], axis=2)  # (N,6,K)
    N = M.shape[0]
    conds = np.zeros(N)
    for i in range(N):
        s = np.linalg.svd(M[i], compute_uv=False)
        conds[i] = s[0] / max(s[-1], 1e-300)
    a, valid = invert_case(fc["b6"], b6, SEL)
    phys = (fc["k"] > 1e-4) & (fc["eps"] > 1e-12) & np.isfinite(fc["b6"]).all(1)
    nw = np.abs(fc["yh"]) > 0.9
    m14 = valid & phys & (conds <= 1e4)
    m13 = valid & phys & (conds <= 1e3)
    amax14 = np.nanmax(np.abs(a[m14])) if m14.any() else float("nan")
    amax13 = np.nanmax(np.abs(a[m13])) if m13.any() else float("nan")
    print(f"{ro:>6g} {N:>4} {int((conds>1e3).sum()):>6} {int((conds>1e4).sum()):>6} "
          f"{int((conds>1e6).sum()):>6} {int(nw.sum()):>7} {int((nw&m14).sum()):>9} "
          f"{int((nw&m13).sum()):>9} {amax14:>15.3g} {amax13:>15.3g}")

# cond 分位数 (全 5 工况合并)
allc = []
for ro in RO_CASES:
    fc = load_case(ro, variant="vort")
    ke = (fc["k"] / fc["eps"])[:, None, None]
    Ss = ke * fc["S"]; Wv = ke * fc["Ot"]
    b6 = {}
    for n, T in pope_tensors(Ss, Wv):
        b6[("T1" if n == "T1" else n + "v")] = sym6(T)
    M = np.stack([b6[n] for n in SEL], axis=2)
    for i in range(M.shape[0]):
        s = np.linalg.svd(M[i], compute_uv=False)
        allc.append(s[0] / max(s[-1], 1e-300))
allc = np.array(allc)
print("\ncond 分位数:", {q: f"{np.percentile(allc, q):.3g}" for q in (50, 75, 90, 95, 99, 100)})
