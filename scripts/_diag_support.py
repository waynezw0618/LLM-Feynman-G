# 诊断: Ro=75 工况下, 19 个 Pope 候选基对 6 个 b_ij 分量的支撑 (哪些基携带哪些分量)
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from rc_asm_prep import load_case, COMP6, sym6
from rc_pope_prep import pope_tensors, invariants

fc = load_case(75.0, variant="vort")
ke = (fc["k"] / fc["eps"])[:, None, None]
Ss = ke * fc["S"]
Wv = ke * fc["Ot"]
We = fc["Ot"].copy()
We[:, 1, 2] += 2.0 * fc["Om_eff"]
We[:, 2, 1] -= 2.0 * fc["Om_eff"]
We = ke * We

print("DNS b6 各分量的量级 (max|.|, median|.|(y/h<0.3 近壁), value@center):")
yh = fc["yh"]
ic = np.argmin(np.abs(yh))
iw = np.abs(yh) < 0.3
for i, c in enumerate(COMP6):
    v = fc["b6"][:, i]
    print(f"  {c}: max={np.abs(v).max():.4f}  nearwall_med={np.median(np.abs(v[iw])):.4f}  center={v[ic]:+.4f}")

print()
print("各候选基的分量支撑 (max|T_ij 分量| / max|b6 对应分量|, >1% 记为携带):")
names = ["uu", "vv", "ww", "uv", "uw", "vw"]
rows = []
for Wname, Ws in [("v", Wv), ("e", We)]:
        for n, T in pope_tensors(Ss, Ws):
            T6 = sym6(T)
            if n == "T1" and Wname == "e":
                continue
            full = "T1" if n == "T1" else n + Wname
            sup = []
            for i, c in enumerate(COMP6):
                r = np.abs(T6[:, i]).max() / max(np.abs(fc["b6"][:, i]).max(), 1e-12)
                sup.append(r)
            rows.append((full, sup))

hdr = "basis  " + "  ".join(f"{c:>7}" for c in names)
print(hdr)
for full, sup in rows:
    line = f"{full:>6}  " + "  ".join((f"{r:7.3f}" if r > 0.01 else "    .  ") for r in sup)
    print(line)

print()
print("选定 5 基 [T2v,T5v,T4v,T3v,T1] 的 a_m 反演 vs 公式层误差在哪:")
# 逐点反演
sel = ["T2v", "T5v", "T4v", "T3v", "T1"]
fc2 = load_case(75.0, variant="vort")
# 重新构造候选字典
cand = {}
for Wname, Ws in [("v", Wv), ("e", We)]:
    for n, T in pope_tensors(Ss, Ws):
        key = "T1" if n == "T1" else n + Wname
        if key not in cand:
            cand[key] = sym6(T)
M = np.stack([np.stack([cand[s][:, i] for s in sel], axis=1) for i in range(6)], axis=1)
# M: (N, 6 comp, 5 bases) -> 逐点 lstsq
N = M.shape[0]
a = np.zeros((N, 5))
res = np.zeros(N)
for i in range(N):
    A = M[i]           # (6 comp, 5 bases)
    b = fc["b6"][i]   # (6,)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a[i] = sol
    res[i] = np.linalg.norm(A @ sol - b) / max(np.linalg.norm(b), 1e-12)
print(f"  逐点反演 rel_resid: med={np.median(res):.2e} p90={np.percentile(res,90):.2e}  (基底层几乎精确)")
print()
print("  逐点条件数 cond(A) 与 a_m 爆炸点定位:")
cond = np.array([np.linalg.cond(M[i]) for i in range(N)])
print(f"  cond: med={np.median(cond):.3g} p90={np.percentile(cond,90):.3g} max={cond.max():.3g}")
for j, s in enumerate(sel):
    jb = np.argmax(np.abs(a[:, j]))
    print(f"  a_{s}: max|a|={np.abs(a[jb,j]):.3g} @ y/h={yh[jb]:+.4f} (y+={fc['yp'][jb]:.1f}), 当地cond={cond[jb]:.3g}, top3|a|位置 y/h={np.round(yh[np.argsort(np.abs(a[:,j]))[-3:]],3)}")
print()
print("  各 a_m 的 y 向结构 (按 |a| 归一化后过零次数 + 近壁/中心值):")
for j, s in enumerate(sel):
    aj = a[:, j]
    nrm = np.abs(aj).max()
    sign_changes = int(np.sum(np.diff(np.sign(aj[np.abs(aj) > 0.02 * nrm])) != 0))
    print(f"  a_{s}: max|a|={nrm:.4f}  nearwall_med={np.median(np.abs(aj[iw])):.2e}  "
          f"center={aj[ic]:+.3e}  显著过零次数~{sign_changes}")
