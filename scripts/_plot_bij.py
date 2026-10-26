"""_plot_bij.py — 绘制各向异性张量 b_ij 分量的分布 (DNS 剖面 + Lumley 三角).

b_ij = <u_i u_j> / (2k) - delta_ij/3  (与 rc_asm_prep.load_case 口径一致)
输出:
  outputs/asm/figures/bij_profiles.png   6 分量 x 5 工况 剖面
  outputs/asm/figures/bij_lumley.png     Lumley 不变量三角形 (各向异性状态分布)
"""
from __future__ import annotations
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_prep import load_case, RO_CASES, OUT_DIR

FIG_DIR = os.path.join(OUT_DIR, "figures")
CN6 = ["b11", "b22", "b33", "b12", "b13", "b23"]


def lumley_xy(b):
    """(N,6) b_ij 六分量 -> Lumley 坐标 (x, y).

    II   = -tr(b^2)/2        (>= 0)
    III  = tr(b^3)/3 = det(b) (可正可负; tr b = 0)
    x = sqrt(II/3),  y = sgn(III) * cbrt(|III|/2)
    """
    B = np.zeros((b.shape[0], 3, 3))
    B[:, 0, 0], B[:, 1, 1], B[:, 2, 2] = b[:, 0], b[:, 1], b[:, 2]
    B[:, 0, 1] = B[:, 1, 0] = b[:, 3]
    B[:, 0, 2] = B[:, 2, 0] = b[:, 4]
    B[:, 1, 2] = B[:, 2, 1] = b[:, 5]
    II = 0.5 * np.einsum("nij,nij->n", B, B)   # = tr(b^2)/2 >= 0
    III = np.einsum("nij,njk,nki->n", B, B, B) / 3.0
    x = np.sqrt(np.maximum(II, 0) / 3.0)
    y = np.sign(III) * np.cbrt(np.abs(III) / 2.0)
    return x, y


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    cases = [load_case(ro, variant="ext") for ro in RO_CASES]

    # ---- 图1: 6 分量剖面 (每个分量一张子图, 5 条 Ro 曲线) ----
    cols = plt.cm.viridis(np.linspace(0, 1, len(cases)))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), sharey=True)
    axes = axes.ravel()
    for i in range(6):
        ax = axes[i]
        for j, fc in enumerate(cases):
            ax.plot(fc["b6"][:, i], fc["yh"], color=cols[j], lw=1.4,
                    label=f"$Ro_\\tau$={fc['ro']:g}")
        ax.set_title(f"${CN6[i]}$", fontsize=12)
        ax.set_xlabel("$b_{ij}$")
        ax.grid(alpha=0.3)
        ax.axvline(0, color="gray", lw=0.6, ls=":")
    axes[0].set_ylabel("$y/h$")
    axes[3].set_ylabel("$y/h$")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Reynolds-stress anisotropy tensor components "
                 "$b_{ij}=\\langle u_i u_j\\rangle/(2k)-\\delta_{ij}/3$ "
                 "(Yang & Wang 2018 DNS, $Re_\\tau=180$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f1 = os.path.join(FIG_DIR, "bij_profiles.png")
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # ---- 图2: Lumley 不变量三角形 (各向异性状态分布) ----
    # 理论边界 (b 本征值满足 sum=0, lam_min >= -1/3):
    #   轴对称 rod  (l2=l3=-b, l1=2b):   y =  x,  x in [0, 1/3] -> 1C 顶点 (1/3, 1/3)
    #   轴对称 disk (l1=l2= b, l3=-2b):   y = -x,  x in [0, 1/6] -> 2C 顶点 (1/6, -1/6)
    #   2C 边界 (l3 = -1/3):              l1=1/6+t, l2=1/6-t, t in [0, 1/2] 参数曲线
    fig, axes2 = plt.subplots(1, 2, figsize=(11.5, 5.4))

    t = np.linspace(0, 0.5, 300)                     # 2C 边界参数化
    II_2c = 0.5 * ((1 / 6 + t) ** 2 + (1 / 6 - t) ** 2 + 1 / 9)
    III_2c = (1 / 6 + t) * (1 / 6 - t) * (-1 / 3)
    x2c = np.sqrt(II_2c / 3.0)
    y2c = np.sign(III_2c) * np.cbrt(np.abs(III_2c) / 2.0)

    for fc in cases:
        fc["_lx"], fc["_ly"] = lumley_xy(fc["b6"])

    for ax, mode in zip(axes2, ("Ro", "y")):
        ax.plot([0, 1 / 3], [0, 1 / 3], "k-", lw=1.0)   # rod 轴对称
        ax.plot([0, 1 / 6], [0, -1 / 6], "k-", lw=1.0)  # disk 轴对称
        ax.plot(x2c, y2c, "k-", lw=1.0)                 # 2C 边界
        ax.plot(0, 0, "k^", ms=5)
        ax.annotate("iso", (0, 0), xytext=(0.005, 0.012), fontsize=8)
        ax.annotate("1C", (1 / 3, 1 / 3), xytext=(0.305, 0.315), fontsize=8)
        ax.annotate("2C", (1 / 6, -1 / 6), xytext=(0.15, -0.185), fontsize=8)
        if mode == "Ro":
            for j, fc in enumerate(cases):
                ax.plot(fc["_lx"], fc["_ly"], ".", ms=2.2, color=cols[j],
                        alpha=0.6, label=f"$Ro_\\tau$={fc['ro']:g}")
            ax.legend(fontsize=8, loc="upper left")
        else:
            allx = np.concatenate([fc["_lx"] for fc in cases])
            ally = np.concatenate([fc["_ly"] for fc in cases])
            allyh = np.concatenate([np.abs(fc["yh"]) for fc in cases])
            sc = ax.scatter(allx, ally, c=allyh, s=3, cmap="coolwarm", alpha=0.6)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85)
            cb.set_label("$|y|/h$")
        ax.set_xlim(-0.02, 0.36)
        ax.set_ylim(-0.21, 0.36)
        ax.set_xlabel(r"$x=\sqrt{II/3},\ II=\mathrm{tr}(b^2)/2$")
        ax.set_ylabel(r"$y=\mathrm{sgn}(III)\,|III/2|^{1/3}$")
        ax.grid(alpha=0.25)
    axes2[0].set_title("colored by $Ro_\\tau$")
    axes2[1].set_title("colored by wall distance $|y|/h$")
    fig.suptitle("Lumley triangle: anisotropy state distribution", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    f2 = os.path.join(FIG_DIR, "bij_lumley.png")
    fig.savefig(f2, dpi=140)
    plt.close(fig)

    # ---- 摘要表 (打印) ----
    print(f"{'Ro':>5} " + " ".join(f"{c:>16}" for c in CN6))
    for fc in cases:
        rng = []
        for i in range(6):
            v = fc["b6"][np.isfinite(fc["b6"][:, i]), i]
            rng.append(f"[{v.min():+.3f},{v.max():+.3f}]")
        print(f"{fc['ro']:>5g} " + " ".join(f"{r:>16}" for r in rng))
    print(f"[bij] figures -> {f1}")
    print(f"[bij]         -> {f2}")


if __name__ == "__main__":
    main()
