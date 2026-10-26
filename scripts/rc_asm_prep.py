"""rc_asm_prep.py — 旋转槽道流 ASM 数据管线 (阶段1: 解析+校核+基底反演+数据集).

数据: Yang & Wang (2018) JFM 838:658-689 DNS (Re_tau=180, Ro_tau=7.5/15/30/75/150).
参考系: x1 流向(旋转轴), x2 壁法向, x3 展向; Ro_tau = 2*Omega*h/u_tau.
扩展内禀旋转张量 Omega_tilde = W + 2*eps*Omega (黄于宁/马晖扬 2008 式(19)):
  对本流动唯一非平凡修正 Omega_tilde_23 = 0.5*dW/dy + 2*Omega.

方案B(主): b_ij = sum_m a_m T_ij^(m)(S*, Om*), S*=(k/eps)S, Om*=(k/eps)Om_tilde.
  先逐点最小二乘反演 a_m(y), 供后续符号回归发现 a_m(不变量).
方案A(对照): 逐分量 b_ij 直接回归 (特征: S*/Om* 分量 + 不变量).

用法: python scripts/rc_asm_prep.py [--variant ext|vort] [--out dataset.npz]
  --variant ext  : 扩展内禀旋转 (默认, Ω̃ = W̄ + 2Ω)
  --variant vort : 纯涡量张量 (Ω* = W̄, 不加帧旋转项; 旋转信息仅经 TG 二次流的 dW/dy 进入)
输出: outputs/asm/dataset.npz (或 --out), prep_report.json, figures/prep_*.png
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(REPO, "outputs", "asm")
FIG_DIR = os.path.join(OUT_DIR, "figures")

RE_TAU = 180.0
H_PLUS = 180.0
RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
FILES = {7.5: "Retau180Ro7.5.txt", 15.0: "Retau180Ro15.txt", 30.0: "Retau180Ro30.txt",
         75.0: "Retau180Ro75.txt", 150.0: "Retau180Ro150.txt"}
COMPS = ["uu", "vv", "ww", "uv", "uw", "vw"]
TERMS = ["produc", "rot", "p-strain", "dissip", "t-diff", "p-diff", "v-diff", "res"]
SYM6 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
COMP6 = ["uu", "vv", "ww", "uv", "uw", "vw"]


def _bud(d, comp, term):
    return d[:, 10 + COMPS.index(comp) * 8 + TERMS.index(term)]


def load_case(ro, variant="ext"):
    path = os.path.join(DATA_DIR, FILES[ro])
    d = np.loadtxt(path, comments="%")
    assert d.shape[1] == 58, f"{path}: cols={d.shape[1]} != 58"
    yp = d[:, 0] * H_PLUS
    U, W = d[:, 1], d[:, 2]
    R = {c: d[:, 3 + i] for i, c in enumerate(COMPS)}
    k = d[:, 9]
    tke_err = float(np.max(np.abs(k - 0.5 * (R["uu"] + R["vv"] + R["ww"]))
                           / np.maximum(k, 1e-12)))
    A = np.gradient(U, yp)   # dU+/dy+
    B = np.gradient(W, yp)   # dW+/dy+
    m = np.abs(_bud(d, "uu", "produc")) > 1e-12
    puu_corr = float(np.corrcoef((-2.0 * R["uv"] * A)[m], _bud(d, "uu", "produc")[m])[0, 1])
    puu_ratio = float(np.median((-2.0 * R["uv"] * A)[m] / _bud(d, "uu", "produc")[m]))
    # rot_vv = +4*Om*<vw> (科氏项), 用相关判符号
    Om_p = ro / (2.0 * H_PLUS)
    rvv = _bud(d, "vv", "rot")
    c_pos = float(np.corrcoef(rvv[m], (4.0 * Om_p * R["vw"])[m])[0, 1])
    c_neg = float(np.corrcoef(rvv[m], (-4.0 * Om_p * R["vw"])[m])[0, 1])
    Om_eff = Om_p if c_pos > c_neg else -Om_p
    # eps: 假设 dissip_ii = -2*eps_ii, 用 k 收支闭合校核系数
    dsum = sum(_bud(d, c, "dissip") for c in ("uu", "vv", "ww"))
    Psum = sum(_bud(d, c, "produc") for c in ("uu", "vv", "ww"))
    Tsum = sum(_bud(d, c, t) for c in ("uu", "vv", "ww")
               for t in ("t-diff", "p-diff", "v-diff"))
    mm = np.abs(dsum) > 1e-10
    lam = float(np.median(-(Psum + Tsum)[mm] / dsum[mm]))
    eps_raw = np.maximum(-0.5 * dsum, 1e-30)
    eps = eps_raw if 0.7 < lam < 1.3 else lam * eps_raw
    eps_note = ("dissip_ii=-2*eps_ii 成立" if 0.7 < lam < 1.3
                else f"按 k 收支修正 lambda={lam:.3f}")
    N = d.shape[0]
    S = np.zeros((N, 3, 3)); S[:, 0, 1] = S[:, 1, 0] = 0.5 * A; S[:, 1, 2] = S[:, 2, 1] = 0.5 * B
    W = np.zeros((N, 3, 3)); W[:, 0, 1] = 0.5 * A; W[:, 1, 0] = -0.5 * A
    W[:, 1, 2] = 0.5 * B; W[:, 2, 1] = -0.5 * B
    Ot = W.copy()
    if variant == "ext":
        Ot[:, 1, 2] += 2.0 * Om_eff; Ot[:, 2, 1] -= 2.0 * Om_eff
    b6 = np.stack([R[c] for c in COMP6], axis=1) / (2.0 * k)[:, None] \
        - np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])[None, :]
    return dict(ro=ro, yh=d[:, 0], yp=yp, U=U, W=W, R=R, k=k, eps=eps, S=S, Ot=Ot,
                b6=b6, Om_eff=Om_eff,
                checks=dict(n_pts=int(N), tke_err=tke_err, puu_corr=puu_corr,
                            puu_ratio=puu_ratio, rot_cpos=c_pos, rot_cneg=c_neg,
                            eps_lambda=lam, eps_note=eps_note))


def derive_stars(fc):
    ke = (fc["k"] / fc["eps"])[:, None, None]
    return ke * fc["S"], ke * fc["Ot"]


def sym6(T):
    return np.stack([T[:, i, j] for (i, j) in SYM6], axis=1)


def tensor_basis(Ss, Os):
    """候选对称无迹基张量 (Pope/Gatski-Speziale 型, 含 Omega_tilde)."""
    I = np.eye(3)[None, :, :]
    tr = lambda M: np.einsum("nii->n", M)
    tl = lambda M: M - (tr(M) / 3.0)[:, None, None] * I
    S2 = Ss @ Ss; O2 = Os @ Os
    return [
        ("S", Ss.copy()),
        ("SOm", Ss @ Os - Os @ Ss),
        ("S2", tl(S2)),
        ("Om2", tl(O2)),
        ("Om2S", tl(O2 @ Ss + Ss @ O2)),
        ("OSO", tl(Os @ Ss @ Os)),
        ("Om2S2", tl(O2 @ S2 + S2 @ O2)),
    ]


def select_basis(basis6, tol=0.02):
    """全局 Gram-Schmidt 秩选择 (仅作参考报告; 实际基底固定用 SEL_BASIS)."""
    sel, vecs = [], []
    for name, V in basis6:
        v = V.reshape(-1).astype(float)
        n = np.linalg.norm(v)
        if n < 1e-14:
            continue
        w = v / n
        for u in vecs:
            w = w - (u @ w) * u
        if np.linalg.norm(w) > tol:
            vecs.append(w / np.linalg.norm(w))
            sel.append(name)
    return sel


# 固定基底层 (Gatski-Speziale EASM 结构): S=线性涡粘项, SOm=旋转线性修正,
# S2/Om2=二次, Om2S=三次. 数值验证 (见 _probe_rank2.py): 该 5 张量在本流动族
# 逐点秩=5 (对称无迹空间张满), a_m 逐点唯一; S 亦可由其余 5 张量表出 (备选).
SEL_BASIS = ["S", "SOm", "S2", "Om2", "Om2S"]


def pointwise_rank(basis6, sel):
    """逐点秩 (6xM SVD, 阈值相对第一奇异值 1e-10)."""
    M = np.stack([basis6[n] for n in sel], axis=2)
    r = np.zeros(M.shape[0], dtype=int)
    for i in range(M.shape[0]):
        s = np.linalg.svd(M[i], compute_uv=False)
        r[i] = int(np.sum(s / max(s[0], 1e-300) > 1e-10))
    return r


def pointwise_cond(basis6, sel):
    """逐点条件数 (6xM 堆叠奇异值比)."""
    M = np.stack([basis6[n] for n in sel], axis=2)
    c = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        s = np.linalg.svd(M[i], compute_uv=False)
        c[i] = s[0] / max(s[-1], 1e-300)
    return c


def invert_am(b6, basis6, sel):
    M = np.stack([basis6[n] for n in sel], axis=2)  # (N,6,M)
    N = b6.shape[0]
    a = np.zeros((N, len(sel))); res = np.zeros(N)
    for i in range(N):
        sol, *_ = np.linalg.lstsq(M[i], b6[i], rcond=None)
        a[i] = sol
        res[i] = np.linalg.norm(M[i] @ sol - b6[i]) / max(np.linalg.norm(b6[i]), 1e-12)
    return a, res


def invariants(Ss, Os):
    S2 = Ss @ Ss; O2 = Os @ Os
    e = lambda M: np.einsum("nii->n", M)
    return {
        "eta2": e(S2),
        "zeta2": -e(O2),
        "s3": e(Ss @ Ss @ Ss),
        "so2": np.einsum("nij,njk,nki->n", Ss, Os, Os),
        "s2o2": np.einsum("nij,nij->n", S2, O2),         # tr(S*^2 Om*^2)
    }


FEAT_B = ["eta2", "zeta2", "s3", "so2", "s2o2"]
FEAT_A = ["s12", "s23", "o12", "o23", "eta2", "zeta2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["ext", "vort"], default="ext")
    ap.add_argument("--out", default=None, help="npz 文件名 (默认 dataset.npz / dataset_vort.npz)")
    ap.add_argument("--cond_max", type=float, default=None,
                    help="剔除逐点基底条件数超过该值的点 (如 1e3), 写入 valid 掩码")
    args = ap.parse_args()
    suffix = "" if args.variant == "ext" else "_vort"
    out_npz = args.out or f"dataset{suffix}.npz"
    os.makedirs(FIG_DIR, exist_ok=True)
    cases = []
    report = {"cases": {}, "variant": args.variant}
    for ro in RO_CASES:
        fc = load_case(ro, variant=args.variant)
        Ss, Os = derive_stars(fc)
        fc["Ss"], fc["Os"] = Ss, Os
        fc["_basis6"] = {n: sym6(T) for n, T in tensor_basis(Ss, Os)}
        cases.append(fc)
        ck = fc["checks"]
        report["cases"][str(ro)] = ck
        print(f"[prep] Ro={ro:>5}: {ck['n_pts']}pts tke_err={ck['tke_err']:.1e} "
              f"puu={ck['puu_corr']:.3f}/{ck['puu_ratio']:.3f} "
              f"rot(+/-)={ck['rot_cpos']:.3f}/{ck['rot_cneg']:.3f} "
              f"eps_lam={ck['eps_lambda']:.3f} [{ck['eps_note']}]")

    names = [n for n, _ in tensor_basis(cases[0]["Ss"], cases[0]["Os"])]
    stacked = {n: np.concatenate([fc["_basis6"][n] for fc in cases]) for n in names}
    sel_ref = select_basis([(n, stacked[n]) for n in names])
    sel = SEL_BASIS
    print(f"[prep] 全局秩参考(过参数化, 仅报告): {sel_ref}")
    print(f"[prep] 固定基底层 (Gatski-Speziale 结构, 逐点秩=5): {sel}")
    report["selected_basis"] = sel
    report["global_rank_basis"] = sel_ref

    for fc in cases:
        fc["_a"], fc["_res"] = invert_am(fc["b6"], fc["_basis6"], sel)
        fc["_rank"] = pointwise_rank(fc["_basis6"], sel)
        fc["_cond"] = pointwise_cond(fc["_basis6"], sel)
        u, ct = np.unique(fc["_rank"], return_counts=True)
        print(f"[prep] Ro={fc['ro']:>5}: 逐点秩 {dict(zip(u.tolist(), ct.tolist()))}, "
              f"反演残差 med={np.median(fc['_res']):.2e} p90={np.percentile(fc['_res'], 90):.2e}")

    save = {"sel_basis": np.array(sel), "featB_keys": np.array(FEAT_B),
            "featA_keys": np.array(FEAT_A)}
    for i, fc in enumerate(cases):
        p = f"c{i}_"
        inv = invariants(fc["Ss"], fc["Os"])
        valid = ((fc["k"] > 1e-4) & (fc["eps"] > 1e-12)
                 & np.isfinite(fc["b6"]).all(1) & np.isfinite(fc["_a"]).all(1)
                 & (fc["_rank"] == len(sel)))
        if args.cond_max:
            valid &= (fc["_cond"] <= args.cond_max)
            print(f"[prep] Ro={fc['ro']:>5}: cond<={args.cond_max:g} 保留 "
                  f"{int(valid.sum())}/{len(valid)} 点")
        featB = np.stack([inv[k2] for k2 in FEAT_B], axis=1)
        featA = np.stack([fc["Ss"][:, 0, 1], fc["Ss"][:, 1, 2],
                          fc["Os"][:, 0, 1], fc["Os"][:, 1, 2],
                          inv["eta2"], inv["zeta2"]], axis=1)
        save.update({
            p + "ro": fc["ro"], p + "yh": fc["yh"], p + "yp": fc["yp"],
            p + "k": fc["k"], p + "eps": fc["eps"], p + "valid": valid,
            p + "b6": fc["b6"],
            p + "R6": np.stack([fc["R"][c] for c in COMP6], 1),
            p + "featB": featB, p + "featA": featA,
            p + "a": fc["_a"], p + "invres": fc["_res"],
            p + "S6": sym6(fc["Ss"]), p + "O6": sym6(fc["Ot"]),
        })
        for n in sel:
            save[f"{p}T_{n}"] = fc["_basis6"][n]
    np.savez_compressed(os.path.join(OUT_DIR, out_npz), **save)
    with open(os.path.join(OUT_DIR, f"prep_report{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[prep] -> {os.path.join(OUT_DIR, out_npz)}")
    _plots(cases, sel, suffix)


def _plots(cases, sel, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cn = ["b11", "b22", "b33", "b12", "b13", "b23"]
    fig, axes = plt.subplots(6, len(cases), figsize=(3.1 * len(cases), 13), sharey=True)
    for j, fc in enumerate(cases):
        bhat = np.zeros_like(fc["b6"])
        for mi, n in enumerate(sel):
            bhat += fc["_a"][:, mi][:, None] * fc["_basis6"][n]
        for i in range(6):
            ax = axes[i, j]
            ax.plot(fc["b6"][:, i], fc["yh"], "o", ms=2, color="k", label="DNS")
            ax.plot(bhat[:, i], fc["yh"], "-", lw=1, color="crimson",
                    label="basis($a_m$)")
            if i == 0:
                ax.set_title(f"$Ro_\\tau={fc['ro']:g}$")
            if j == 0:
                ax.set_ylabel(f"${cn[i]}$\n$y/h$")
            if i == 5:
                ax.set_xlabel("value")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Expressibility: DNS anisotropy vs tensor-basis reconstruction (pointwise $a_m$)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"prep_basis_expressibility{suffix}.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(sel), figsize=(2.9 * len(sel), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    cols = plt.cm.viridis(np.linspace(0, 1, len(cases)))
    for mi, n in enumerate(sel):
        for j, fc in enumerate(cases):
            axes[mi].plot(fc["_a"][:, mi], fc["yh"], color=cols[j],
                          label=f"$Ro_\\tau$={fc['ro']:g}")
        axes[mi].set_title(f"$a_{{{n}}}$")
        axes[mi].grid(alpha=0.3); axes[mi].axvline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("$y/h$")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"prep_am_profiles{suffix}.png"), dpi=130)
    plt.close(fig)
    print(f"[prep] figures -> prep_*{suffix}.png")


if __name__ == "__main__":
    main()
