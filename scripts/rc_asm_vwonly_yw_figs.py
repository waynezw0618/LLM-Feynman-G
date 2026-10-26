# -*- coding: utf-8 -*-
"""vw-only yw 轮 (12/12) 报告图表生成器.

产出 (outputs/asm/figs_vwonly_yw/):
  fig1_rank_loss_rho.png   — loss_vw vs rho_mean 双口径排名散点 (log x)
  fig2_rho_per_ro.png      — 各 Ro 的逐工况 rho 曲线 (top-5 组)
  fig3_profiles.png        — b23 剖面对比 (truth vs pysr/m vs lgf_go/e, Ro=7.5/30/150)
  fig4_nmse_heat.png       — 12 组 x 5 工况 NMSE 热图
  summary_yw.json          — 全部汇总数值 (报告正文引用)

口径: NMSE 分母 = var_c (正式口径, rc_asm_vw_only.py eval_vw_predictions);
      LGF 组用文本语义求值 (theta 按排序后参数名对位, 已验证 10/11 与管线记录一致,
      lgf_go/m 为 DSL 往返 bug 修正值).
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, r"d:\AI\LLM-Feynman-G\scripts")
import rc_asm_vw_only as R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"d:\AI\LLM-Feynman-G\outputs\asm"
FIGD = os.path.join(BASE, "figs_vwonly_yw")
os.makedirs(FIGD, exist_ok=True)

ENGINES = ["pysr", "lgf_go", "lgf_mg"]
VARS = ["v", "g", "e", "m"]
RO_LIST = [7.5, 15.0, 30.0, 75.0, 150.0]
RO_TAG = ["Ro7.5", "Ro15", "Ro30", "Ro75", "Ro150"]

PY2NP = {"sqrt": np.sqrt, "log": np.log, "exp": np.exp, "tanh": np.tanh,
         "abs": np.abs, "sin": np.sin, "cos": np.cos, "min": np.minimum,
         "max": np.maximum}


def textual_eval(dsl, theta, X, vn):
    names = sorted(set(re.findall(r"\bC\d+\b", dsl)), key=lambda s: int(s[1:]))
    ns = {v: X[:, j] for j, v in enumerate(vn)}
    ns.update({n: float(t) for n, t in zip(names, theta)})
    ns.update(PY2NP)
    return np.asarray(eval(dsl, {"__builtins__": {}}, ns), dtype=float)


def official_metrics(preds, data):
    per_nmse, per_rho = [], []
    for p, c in zip(preds, data):
        y = c["yfit"]
        per_nmse.append(float(np.mean((p - y) ** 2) / c["var_c"]))
        per_rho.append(float(np.corrcoef(p, y)[0, 1]))
    pooled = float(np.corrcoef(np.concatenate(preds),
                               np.concatenate([c["yfit"] for c in data]))[0, 1])
    return per_nmse, per_rho, pooled


GROUPS = {}   # name -> dict
for eng in ENGINES:
    for var in VARS:
        rd = os.path.join(BASE, f"runs_vwonly_yw_{eng}", var)
        rj = os.path.join(rd, "result.json")
        if not os.path.exists(rj):
            continue
        r = json.load(open(rj, encoding="utf-8"))
        data, vn = R.load_vw_data(var, add_yw=True)
        if eng == "pysr":
            b = r["best"]
            pz = np.load(os.path.join(rd, "preds_best.npz"))
            preds = [np.asarray(pz[f"c{i}"], dtype=float) for i in range(len(data))]
            formula = b["sympy"]
            n_par = b.get("complexity", None)
            tag = f"complexity={n_par}"
        else:
            b = min(r["entries"], key=lambda x: x["loss_paramfit"])
            preds = [textual_eval(b["dsl"], b["theta"], c["Xfit"], vn) for c in data]
            formula = b["dsl"]
            n_par = len(set(re.findall(r"\bC\d+\b", b["dsl"])))
            tag = f"n_params={n_par}"
        rec = b["metrics"]["loss_vw"]
        per_nmse, per_rho, pooled = official_metrics(preds, data)
        GROUPS[f"{eng}/{var}"] = dict(
            eng=eng, var=var, rec_loss=rec, nmse=per_nmse, rho=per_rho,
            rho_pool=pooled, elapsed=r.get("elapsed_s"),
            formula=formula, par=tag, preds=preds,
            worst=float(max(per_nmse)), nmse_mean=float(np.mean(per_nmse)),
            rho_mean=float(np.nanmean(per_rho)),
        )

# ---------- summary json ----------
summary = {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in GROUPS.items()}
json.dump(summary, open(os.path.join(FIGD, "summary_yw.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)

rank = sorted(GROUPS.items(), key=lambda kv: kv[1]["nmse_mean"])
print("== verified ranking (NMSE var_c) ==")
for i, (name, g) in enumerate(rank):
    print(f"{i+1:>2}. {name:<10} nmse={g['nmse_mean']:.4f} worst={g['worst']:.3f} "
          f"rho_mean={g['rho_mean']:.4f} rho_pool={g['rho_pool']:.4f} "
          f"rec={g['rec_loss']:.4f} elapsed={g['elapsed']:.0f}s {g['par']}")

# ---------- fig 1: rank scatter loss vs rho ----------
fig, ax = plt.subplots(figsize=(9, 6.5))
colors = {"pysr": "#d62728", "lgf_go": "#1f77b4", "lgf_mg": "#2ca02c"}
for name, g in GROUPS.items():
    ax.scatter(g["nmse_mean"], g["rho_mean"], s=110,
               c=colors[g["eng"]], edgecolors="k", linewidths=0.6, zorder=3)
    ax.annotate(name, (g["nmse_mean"], g["rho_mean"]),
                textcoords="offset points", xytext=(7, 5), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("NMSE (var_c), mean over 5 Ro  [lower is better]")
ax.set_ylabel(r"mean per-case $\rho$ (pred vs truth)")
ax.set_title("vw-only + yw round: NMSE vs shape correlation (12/12 groups)")
ax.axvline(0.1103, color="gray", ls="--", lw=1)
ax.text(0.1103, ax.get_ylim()[0] + 0.0005, " prev-round SOTA 0.110 (no yw)",
        fontsize=8, color="gray", rotation=90, va="bottom")
ax.grid(alpha=0.3)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lbl)
           for lbl, c in [("PySR", colors["pysr"]),
                          ("LGF-go (minimax, ginit-only)", colors["lgf_go"]),
                          ("LGF-mg (minimax, ginit-only)", colors["lgf_mg"])]]
ax.legend(handles=handles, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig1_rank_loss_rho.png"), dpi=150)
plt.close(fig)

# ---------- fig 2: per-Ro rho for top-5 ----------
top5 = [name for name, _ in rank[:5]]
fig, ax = plt.subplots(figsize=(9, 5.5))
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(top5)))
for name, col in zip(top5, cmap):
    ax.plot(RO_LIST, GROUPS[name]["rho"], "o-", color=col, label=name, lw=1.8, ms=6)
ax.set_xscale("log")
ax.set_xticks(RO_LIST)
ax.set_xticklabels([str(x) for x in RO_LIST])
ax.set_xlabel(r"$Ro_\tau$")
ax.set_ylabel(r"per-case $\rho$")
ax.set_title(r"Shape correlation vs $Ro_\tau$ (top-5 groups)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig2_rho_per_ro.png"), dpi=150)
plt.close(fig)

# ---------- fig 3: b23 profiles ----------
if "pysr/m" in GROUPS and "lgf_go/e" in GROUPS:
    picks = [("pysr/m", GROUPS["pysr/m"]), ("lgf_go/e", GROUPS["lgf_go/e"])]
    idx_ro = [0, 2, 4]   # 7.5, 30, 150
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, ir in zip(axes, idx_ro):
        data, _ = R.load_vw_data("m", add_yw=True)
        c = data[ir]
        yh = c["yh"]
        ax.plot(yh, c["yfit"], "k-", lw=2.2, label="DNS truth")
        for (name, g), ls in zip(picks, ["--", ":"]):
            ax.plot(yh, g["preds"][ir], ls, lw=1.8,
                    color="#d62728" if name.startswith("pysr") else "#1f77b4",
                    label=f"{name} (NMSE={g['nmse'][ir]:.3f})")
        ax.set_xlabel(r"$y/h$")
        ax.set_title(rf"$Ro_\tau$={RO_LIST[ir]:g}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$b_{23}=\langle vw\rangle / u_\tau^2$")
    axes[0].legend(fontsize=9)
    fig.suptitle("b23 profile: truth vs top-2 (m-variant data for pysr/m, e-variant for lgf_go/e)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, "fig3_profiles.png"), dpi=150)
    plt.close(fig)

# ---------- fig 4: NMSE heatmap ----------
names_sorted = [n for n, _ in rank]
mat = np.array([GROUPS[n]["nmse"] for n in names_sorted])
fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(np.log10(mat), cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(5))
ax.set_xticklabels([f"Ro{r:g}" for r in RO_LIST])
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                ha="center", va="center", fontsize=8,
                color="k" if 0.01 < v < 1 else "w")
ax.set_title(r"per-case NMSE (var_c), $\log_{10}$ color scale")
fig.colorbar(im, label=r"$\log_{10}$ NMSE")
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig4_nmse_heat.png"), dpi=150)
plt.close(fig)

print("\nfigures written to", FIGD)
for f in sorted(os.listdir(FIGD)):
    print(" ", f, os.path.getsize(os.path.join(FIGD, f)) // 1024, "KB")

# ---------- old round per-Ro comparison ----------
# 旧轮 LGF 直接用管线记录值 (lgf_mg/g 的 DSL 引用 C5 而 theta 仅 4 个, 文本求值器不适用;
# 该组 loss_paramfit 与重编译评估 8 位有效数字一致, 记录可信)
print("\n== old round (no yw) per-Ro NMSE of best groups ==")
for eng, var in [("lgf_mg", "g"), ("pysr", "e"), ("pysr", "m"), ("lgf_go", "v")]:
    rj = os.path.join(BASE, f"runs_vwonly_{eng}", var, "result.json")
    if not os.path.exists(rj):
        print(f"{eng}/{var}: missing")
        continue
    r = json.load(open(rj, encoding="utf-8"))
    if eng == "pysr":
        b = r["best"]
        data, vn = R.load_vw_data(var, add_yw=False)
        pz = np.load(os.path.join(BASE, f"runs_vwonly_{eng}", var, "preds_best.npz"))
        preds = [np.asarray(pz[f"c{i}"], dtype=float) for i in range(len(data))]
        per_nmse, per_rho, pooled = official_metrics(preds, data)
        per_nmse = list(per_nmse)
    else:
        b = min(r["entries"], key=lambda x: x["loss_paramfit"])
        per_nmse = [v["nmse"] for v in b["metrics"]["per_case"].values()]
    print(f"{eng}/{var}: mean={np.mean(per_nmse):.4f} per-ro=" +
          " ".join(f"{v:.3f}" for v in per_nmse))
