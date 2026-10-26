"""快速探查: 7 个候选基张量的逐点秩与最小独立子集."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rc_asm_prep import (load_case, derive_stars, tensor_basis, sym6,
                                 RO_CASES, SYM6)

def ranks(V6):
    """V6: list of (N,6) -> 逐点秩分布 (对 6xM 矩阵做 SVD)."""
    M = np.stack(V6, axis=2)  # (N,6,M)
    r = np.zeros(M.shape[0], dtype=int)
    for i in range(M.shape[0]):
        s = np.linalg.svd(M[i], compute_uv=False)
        s = s / max(s[0], 1e-300)
        r[i] = int(np.sum(s > 1e-10))
    return r

all_basis = []
for ro in RO_CASES:
    fc = load_case(ro)
    Ss, Os = derive_stars(fc)
    tb = tensor_basis(Ss, Os)
    all_basis.append({n: sym6(T) for n, T in tb})

names = list(all_basis[0].keys())
print("候选张量:", names)
for j, b in enumerate(all_basis):
    r = ranks([b[n] for n in names])
    print(f"Ro={RO_CASES[j]:>5}: 逐点秩分布 {dict(zip(*np.unique(r, return_counts=True)))}")

# 贪心选子集: 使逐点秩最大化 (在全部工况点上)
from itertools import combinations
best = None
for k in range(3, 6):
    for combo in combinations(names, k):
        ok = True
        tot = 0
        for b in all_basis:
            r = ranks([b[n] for n in combo])
            tot += np.mean(r)
        if best is None or tot > best[1]:
            best = (combo, tot)
    print(f"k={k}: 最优子集 {best[0]} 平均逐点秩 {best[1]/len(all_basis):.4f}")
