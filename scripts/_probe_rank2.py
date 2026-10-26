"""测试含 S 的 5 张量子集的逐点秩与 S 的可表达性."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rc_asm_prep import (load_case, derive_stars, tensor_basis, sym6, RO_CASES)

all_basis = []
for ro in RO_CASES:
    fc = load_case(ro)
    Ss, Os = derive_stars(fc)
    tb = tensor_basis(Ss, Os)
    all_basis.append({n: sym6(T) for n, T in tb})

def rank_stats(basis, combo):
    rs, resS = [], []
    for b in basis:
        M = np.stack([b[n] for n in combo], axis=2)  # (N,6,M)
        for i in range(M.shape[0]):
            s = np.linalg.svd(M[i], compute_uv=False)
            rs.append(int(np.sum(s / max(s[0], 1e-300) > 1e-10)))
    return np.mean(rs)

def express_S(basis, others):
    """S 能否被 others 逐点线性表出: 逐点 lstsq 残差."""
    out = []
    for b in basis:
        M = np.stack([b[n] for n in others], axis=2)
        r = np.zeros(M.shape[0])
        for i in range(M.shape[0]):
            sol, *_ = np.linalg.lstsq(M[i], b["S"][i], rcond=None)
            r[i] = np.linalg.norm(M[i] @ sol - b["S"][i]) / max(np.linalg.norm(b["S"][i]), 1e-30)
        out.append(np.median(r))
    return out

others = ["SOm", "S2", "Om2", "Om2S", "OSO"]
print("S 被 {SOm,S2,Om2,Om2S,OSO} 逐点表出的中位相对残差 (5 工况):")
print(express_S(all_basis, others))

for combo in [("S","SOm","S2","Om2","Om2S"), ("S","SOm","S2","Om2","OSO"),
              ("S","SOm","S2","Om2","Om2S2"), ("S","SOm","S2","Om2"),
              ("S","SOm","S2","Om2","Om2S","OSO")]:
    print(combo, "平均逐点秩 =", round(rank_stats(all_basis, combo), 4))
