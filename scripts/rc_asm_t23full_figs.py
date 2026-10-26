# -*- coding: utf-8 -*-
"""Figures for T23-full LGF vs PySR comparison -> outputs/asm/t23full_paper/figs/.

fig1: loss stability strip plot (PySR 3 seeds vs LGF 2 arms) + std bars
fig2: champion profile curves (truth vs pred, 5 Ro cases, per method)
fig3: complexity-loss scatter (Pareto view)
fig4: per-case NMSE heatmap across all runs
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\AI\LLM-Feynman-G")
sys.path.insert(0, str(ROOT / "scripts"))
OUTDIR = ROOT / "outputs" / "asm" / "t23full_paper"
FIGDIR = OUTDIR / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["v", "g", "e", "m"]
VAR_LABEL = {"v": "v (W=W\u0304)", "g": "g (W=W\u0304+\u03a9)",
             "e": "e (W=W\u0304+2\u03a9)", "m": "m (43-dim mix)"}
RO_KEYS = ["Ro7.5", "Ro15", "Ro30", "Ro75", "Ro150"]
RO_LBL = ["Ro\u03c4=7.5", "Ro\u03c4=15", "Ro\u03c4=30", "Ro\u03c4=75", "Ro\u03c4=150"]

C_PYSR = "#3b7dd8"
C_LGF = "#d64528"
plt.rcParams.update({"font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 160,
                     "savefig.bbox": "tight"})


def load_summary():
    return json.loads((OUTDIR / "t23full_summary.json").read_text(encoding="utf-8"))


def fig1_stability(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4),
                             gridspec_kw={"width_ratios": [2.1, 1.0]})
    ax = axes[0]
    for iv, v in enumerate(VARIANTS):
        ps = [r["loss_vw"] for r in rows
              if r["method"] == "pysr" and r["variant"] == v]
        lg_mg = [r["loss_vw"] for r in rows
                 if r["method"] == "lgf_mg" and r["variant"] == v]
        lg_go = [r["loss_vw"] for r in rows
                 if r["method"] == "lgf_go" and r["variant"] == v]
        x0 = iv - 0.18
        if ps:
            ax.scatter([x0 + i * 0.045 - 0.045 for i in range(len(ps))], ps,
                       s=26, color=C_PYSR, zorder=3, label="PySR (3 seeds)" if iv == 0 else None)
            m = float(np.mean(ps))
            ax.plot([x0 - 0.09, x0 + 0.09], [m, m], color=C_PYSR, lw=2.2, zorder=2)
        if lg_mg:
            ax.scatter([iv + 0.16], lg_mg, s=52, marker="s", color=C_LGF,
                       zorder=4, label="LGF mg (1 run)" if iv == 0 else None)
        if lg_go:
            ax.scatter([iv + 0.26], lg_go, s=52, marker="^", color=C_LGF,
                       edgecolor="k", linewidths=0.5, zorder=4,
                       label="LGF go (1 run)" if iv == 0 else None)
        lg = lg_mg + lg_go
        if lg:
            m = float(np.mean(lg))
            ax.plot([iv + 0.11, iv + 0.31], [m, m], color=C_LGF, lw=2.2, zorder=2)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([VAR_LABEL[v] for v in VARIANTS])
    ax.set_ylabel("NMSE (loss_vw, 5-case pooled)")
    ax.set_title("(a) Per-run loss: scatter = runs, bar = mean", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    ax2 = axes[1]
    xs, pysr_std, lgf_spread = [], [], []
    for iv, v in enumerate(VARIANTS):
        ps = [r["loss_vw"] for r in rows
              if r["method"] == "pysr" and r["variant"] == v]
        lg = [r["loss_vw"] for r in rows
              if str(r["method"]).startswith("lgf") and r["variant"] == v]
        if len(ps) > 1 and len(lg) > 1:
            xs.append(VARIANTS.index(v))
            pysr_std.append(float(np.std(ps, ddof=1)))
            lgf_spread.append(float(np.std(lg, ddof=1)))
    w = 0.36
    ax2.bar([x - w / 2 for x in xs], pysr_std, width=w, color=C_PYSR,
            label="PySR std (3 seeds)")
    ax2.bar([x + w / 2 for x in xs], lgf_spread, width=w, color=C_LGF,
            label="LGF spread (2 arms)")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([VARIANTS[x] for x in xs])
    ax2.set_ylabel("run-to-run \u03c3 (NMSE)")
    ax2.set_title("(b) Instability", fontsize=9.5)
    ax2.legend(fontsize=7.5)
    fig.suptitle("T\u2082\u2083-augmented pools: PySR seed variance vs LGF arm stability",
                 fontsize=10.5, y=1.02)
    fig.savefig(FIGDIR / "fig1_t23_loss_stability.png")
    plt.close(fig)


def _load_truth(variant):
    """Reuse the experiment's loader to get truth profiles."""
    from rc_asm_vw_t23_full import load_t23_pool
    cases, _ = load_t23_pool(variant)
    return cases


def fig2_champion_profiles(rows):
    best_lgf = min((r for r in rows if str(r["method"]).startswith("lgf")),
                   key=lambda r: r["loss_vw"], default=None)
    best_pysr = min((r for r in rows if r["method"] == "pysr"),
                    key=lambda r: r["loss_vw"], default=None)
    if not best_lgf or not best_pysr:
        print("[fig2] missing champion, skip")
        return
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.4), sharex=True)
    for row, r in enumerate([best_lgf, best_pysr]):
        cases = _load_truth(r["variant"])
        pz = np.load(r["preds_file"])
        for j, ck in enumerate(RO_KEYS):
            ax = axes[row, j]
            c = cases[j]
            truth = c.get("b23", c.get("yfit"))
            ax.plot(c["yh"], truth, "k-", lw=1.6, label="DNS truth")
            ax.plot(c["yh"], pz[f"c{j}"], "--", color=C_LGF if row == 0 else C_PYSR,
                    lw=1.4, label="predicted")
            nm = r["per_case"][ck]
            ax.set_title(f"{RO_LBL[j]}  NMSE={nm:.3f}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{('LGF ' + r['method'][4:]) if row == 0 else 'PySR'}\nb\u2082\u2083", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("y/h")
            ax.tick_params(labelsize=7.5)
    name_l = f"{best_lgf['method']}/{best_lgf['variant']} (loss {best_lgf['loss_vw']:.3f})"
    name_p = f"pysr/{best_pysr['variant']} seed{best_pysr.get('seed', 0)} (loss {best_pysr['loss_vw']:.3f})"
    fig.suptitle(f"Champion profiles \u2014 top: LGF {name_l}; bottom: PySR {name_p}",
                 fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.965), fontsize=8)
    fig.savefig(FIGDIR / "fig2_t23_champion_profiles.png")
    plt.close(fig)


def fig3_pareto(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    offs = {"v": 0.0, "g": 1.0, "e": 2.0, "m": 3.0}
    for v in VARIANTS:
        for meth, col, mk in [("pysr", C_PYSR, "o"), ("lgf_mg", C_LGF, "s"),
                              ("lgf_go", C_LGF, "^")]:
            sel = [r for r in rows if r["method"] == meth and r["variant"] == v]
            if not sel:
                continue
            ax.scatter([offs[v] + r["complexity_nodes"] for r in sel],
                       [r["loss_vw"] for r in sel], s=34, color=col, marker=mk,
                       edgecolor="k", linewidths=0.4, zorder=3,
                       label={"pysr": "PySR", "lgf_mg": "LGF mg",
                              "lgf_go": "LGF go"}[meth] if v == "v" else None)
    for v, xv in offs.items():
        ax.axvline(xv + 0.5, color="gray", lw=0.5, ls=":")
    ax.set_xticks([offs[v] + 10 for v in VARIANTS])
    ax.set_xticklabels(VARIANTS)
    ax.set_xlabel("formula size (node count; bands = pools v/g/e/m)")
    ax.set_ylabel("NMSE (loss_vw)")
    ax.set_title("Size\u2013accuracy: LGF wins at low complexity", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "fig3_t23_pareto.png")
    plt.close(fig)


def fig4_heatmap(rows):
    order = sorted(rows, key=lambda r: (r["method"] != "pysr", r["variant"], r.get("seed", 0)))
    M = np.array([[r["per_case"][k] for k in RO_KEYS] for r in order])
    names = [f"{r['method']}/{r['variant']}" + (f"#s{r['seed']}" if r["method"] == "pysr" else "")
             for r in order]
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(order) + 1.4))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.55)
    ax.set_xticks(range(len(RO_KEYS)))
    ax.set_xticklabels(RO_KEYS, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.4,
                    color="black")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("per-case NMSE", fontsize=8)
    ax.set_title("Per-Ro case NMSE, all runs (sorted: PySR top, LGF bottom)",
                 fontsize=9.5)
    ax.grid(False)
    fig.savefig(FIGDIR / "fig4_t23_percase_heatmap.png")
    plt.close(fig)


def main():
    rows = load_summary()["rows"]
    fig1_stability(rows)
    try:
        fig2_champion_profiles(rows)
    except Exception as e:  # noqa: BLE001
        print(f"[fig2] failed: {e}")
    fig3_pareto(rows)
    fig4_heatmap(rows)
    print(f"-> figs in {FIGDIR}")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"   {f.name}  {f.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
