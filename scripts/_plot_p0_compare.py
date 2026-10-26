# P0 前后对照剖面图: DNS vs 旧 heuristic_pope_y (ypd+病态反演) vs 新 heuristic_pope_wy (yw+valid_fit+加权)
import json
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rc_asm_eval import _decode_dsl, _eval_one
from rc_asm_prep import RO_CASES, COMP6

OUT = os.path.join(REPO, "outputs", "asm")
FIG = os.path.join(OUT, "figures")

PAIRS = [
    ("heuristic_pope_wy", "dataset_pope_wy.npz", "P0-fixed (yw+valid_fit+weights)", "crimson"),
    ("heuristic_pope_y", "dataset_pope_y.npz", "old (ypd, ill-cond a_m)", "dodgerblue"),
]


def predict_all(tag, dsname):
    ds = np.load(os.path.join(OUT, dsname))
    sel = [str(x) for x in ds["sel_basis"]]
    keys = [str(x) for x in ds["featB_keys"]]
    summary = json.load(open(os.path.join(OUT, f"runs_{tag}", "summary_B.json"),
                             encoding="utf-8"))
    out = {}
    for ci in range(len(RO_CASES)):
        p = f"c{ci}_"; v = ds[p + "valid"]
        Tb = np.stack([ds[p + "T_" + m][v] for m in sel], axis=1)
        rec = np.zeros((int(v.sum()), 6)); ok = True
        for tgt in sel:
            tinfo = summary["targets"].get(tgt)
            if tinfo is None:
                ok = False; break
            f = _decode_dsl(tinfo["dsl"], keys)
            if f is None:
                ok = False; break
            yp = _eval_one(f, ds[p + "featB"][v], np.asarray(tinfo["theta"], float), ci)
            if yp is None or not np.all(np.isfinite(yp)):
                ok = False; break
            rec += yp[:, None] * Tb[:, sel.index(tgt), :]
        out[ci] = (rec if ok else None, ds[p + "b6"][v], ds[p + "yh"][v], v)
    return out


def main():
    warnings.filterwarnings("ignore")
    preds = {tag: predict_all(tag, dsn) for tag, dsn, _, _ in PAIRS}
    for ci, ro in enumerate(RO_CASES):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharey=True)
        for k, comp in enumerate(COMP6):
            ax = axes[k // 3][k % 3]
            ref = preds[PAIRS[0][0]][ci]
            dns, yh = ref[1][:, k], ref[2]
            ax.plot(dns, yh, "k-", lw=2.4, label="DNS")
            lo, hi = float(np.min(dns)), float(np.max(dns))
            span = max(hi - lo, 0.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(lo - 0.2 * span, hi + 0.2 * span)
            for tag, _, lbl, col in PAIRS:
                rec = preds[tag][ci][0]
                if rec is not None:
                    ax.plot(rec[:, k], yh, "--", lw=1.3, color=col, label=lbl, alpha=0.9)
            ax.axhline(0.9, color="gray", lw=0.6, ls=":")
            ax.axhline(-0.9, color="gray", lw=0.6, ls=":")
            ax.set_title(f"<{comp}>+")
            if k % 3 == 0:
                ax.set_ylabel("y/h")
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=8, loc="best")
        fig.suptitle(f"Ro_tau={ro:g}: DNS vs old (ypd) vs P0-fixed (yw) ASM reconstruction"
                     f"   [dotted: near-wall |y/h|>0.9]")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"Ro{ro:g}_P0_before_after.png"), dpi=125)
        plt.close(fig)
        print(f"[fig] Ro{ro:g} -> Ro{ro:g}_P0_before_after.png")


if __name__ == "__main__":
    main()
