# -*- coding: utf-8 -*-
"""vw-only V4 轮 (GLM 5.3 + L1 五项修复) 报告图表生成器.

产出 (outputs/asm/figs_vwonly_v4/):
  fig1_rank_loss_rho.png   — NMSE vs rho 双口径排名散点 (log x)
  fig2_bar_per_ro.png      — 12 组 x 5 Ro 逐工况 NMSE 分组柱状图 (log y)
  fig3_rho_per_ro.png      — top-5 组逐工况 rho 曲线
  fig4_profiles.png        — 每方法最优组 b23 剖面 (Ro=7.5/30/150)
  fig5_history.png         — LGF 8 组 best_loss 训练历史 (vs elapsed s)
  fig6_nmse_heat.png       — 12 组 x 5 工况 NMSE 热图
  summary_v4.json          — 全部汇总数值 (报告正文引用)

口径: NMSE 分母 = var_c (正式口径); theta 配对 = sorted(C# names) ↔ theta
     (V4 保名+按名排序后与 params() 顺序严格一致, 无需文本重验).
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, r"d:\AI\LLM-Feynman-G\scripts")
import rc_asm_vw_only as R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"d:\AI\LLM-Feynman-G\outputs\asm"
FIGD = os.path.join(BASE, "figs_vwonly_v4")
os.makedirs(FIGD, exist_ok=True)

ENGINES = ["pysr", "lgf_go", "lgf_mg"]
VARS = ["v", "g", "e", "m"]
RO_LIST = [7.5, 15.0, 30.0, 75.0, 150.0]
# V4b: LGF 用修复后 tag (_v4b), PySR 沿用 _v4 (PySR 路径不受 LGF 修复影响)
DIR_TAG = {"pysr": "v4", "lgf_go": "v4b", "lgf_mg": "v4b"}

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


# ================= 汇总 12 组 =================
GROUPS = {}
for eng in ENGINES:
    for var in VARS:
        rd = os.path.join(BASE, f"runs_vwonly_{DIR_TAG[eng]}_{eng}", var)
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
        per_nmse, per_rho, pooled = official_metrics(preds, data)
        GROUPS[f"{eng}/{var}"] = dict(
            eng=eng, var=var, rec_loss=b["metrics"]["loss_vw"],
            nmse=per_nmse, rho=per_rho, rho_pool=pooled,
            elapsed=r.get("elapsed_s"), formula=formula, par=tag,
            preds=preds, worst=float(max(per_nmse)),
            nmse_mean=float(np.mean(per_nmse)),
            rho_mean=float(np.nanmean(per_rho)),
        )

n_done = len(GROUPS)
print(f"[v4-figs] {n_done}/12 groups loaded")
if n_done == 0:
    sys.exit("no result.json yet")

summary = {k: {kk: vv for kk, vv in v.items() if kk != "preds"}
           for k, v in GROUPS.items()}
json.dump(summary, open(os.path.join(FIGD, "summary_v4.json"), "w",
                        encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)

rank = sorted(GROUPS.items(), key=lambda kv: kv[1]["nmse_mean"])
print("== verified ranking (NMSE var_c) ==")
for i, (name, g) in enumerate(rank):
    print(f"{i+1:>2}. {name:<10} nmse={g['nmse_mean']:.4f} worst={g['worst']:.3f} "
          f"rho_mean={g['rho_mean']:.4f} rho_pool={g['rho_pool']:.4f} "
          f"rec={g['rec_loss']:.4f} elapsed={g['elapsed']:.0f}s {g['par']}")

colors = {"pysr": "#d62728", "lgf_go": "#1f77b4", "lgf_mg": "#2ca02c"}

# ---------- fig 1: rank scatter ----------
fig, ax = plt.subplots(figsize=(8.2, 6.2))
pts = {}
for name, g in GROUPS.items():
    key = (round(g["nmse_mean"], 6), round(g["rho_mean"], 6))
    pts.setdefault(key, []).append(name)
    ax.scatter(g["nmse_mean"], g["rho_mean"], s=110,
               c=colors[g["eng"]], edgecolors="k", linewidths=0.6, zorder=3)
for (x, y), names in pts.items():
    ax.annotate(", ".join(names),
                (x, y), textcoords="offset points",
                xytext=(5, 5), fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
ax.set_xscale("log")
ax.set_xlabel("NMSE (var_c), mean over 5 Ro  [lower is better]")
ax.set_ylabel(r"mean per-case $\rho$ (pred vs truth)")
ax.set_title(f"vw-only V4: NMSE vs shape correlation ({n_done}/12)")
# remove prev-round reference line
xmin, xmax = min(g["nmse_mean"] for g in GROUPS.values()), max(g["nmse_mean"] for g in GROUPS.values())
ax.set_xlim(xmin * 0.85, xmax * 1.15)
ax.grid(alpha=0.3)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lbl)
           for lbl, c in [("PySR", colors["pysr"]),
                          ("LGF-g (GLM-5.3, ginit-only)", colors["lgf_go"]),
                          ("LGF-m (GLM-5.3, full)", colors["lgf_mg"])]]
ax.legend(handles=handles, loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig1_rank_loss_rho.png"), dpi=150)
plt.close(fig)

# ---------- fig 2: per-Ro NMSE grouped bars + mean marker ----------
names_sorted = [n for n, _ in rank]
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(RO_LIST))
nw = 0.8 / len(names_sorted)
for k, name in enumerate(names_sorted):
    g = GROUPS[name]
    ax.bar(x + (k - len(names_sorted) / 2) * nw + nw / 2, g["nmse"], width=nw * 0.92,
           color=colors[g["eng"]], alpha=0.85, edgecolor="k", linewidth=0.3,
           label=name)
ax.set_yscale("log")
ax.set_xticks(list(range(len(RO_LIST))))
ax.set_xticklabels([f"Ro{r:g}" for r in RO_LIST])
ax.set_ylabel("per-case NMSE (var_c)  [log scale]")
ax.set_title("Per-Rotation NMSE by group (sorted by mean NMSE)")
ax.legend(fontsize=7.5, ncol=4, loc="upper left")
ax.grid(alpha=0.3, axis="y", ls=":")
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig2_bar_per_ro.png"), dpi=150)
plt.close(fig)

# ---------- fig 3: per-Ro rho for top-5 ----------
top5 = [name for name, _ in rank[:5]]
fig, ax = plt.subplots(figsize=(9, 5.5))
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(top5))))
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
fig.savefig(os.path.join(FIGD, "fig3_rho_per_ro.png"), dpi=150)
plt.close(fig)

# ---------- fig 4: profiles (best of each method) ----------
best_of = {}
for eng in ENGINES:
    cands = [(n, g) for n, g in GROUPS.items() if g["eng"] == eng]
    if cands:
        best_of[eng] = min(cands, key=lambda kv: kv[1]["nmse_mean"])
if len(best_of) >= 2:
    idx_ro = [0, 2, 4]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    lss = {"pysr": "--", "lgf_go": ":", "lgf_mg": "-."}
    for ax, ir in zip(axes, idx_ro):
        # 用最优组各自 variant 的数据画 truth (不同 variant 特征池不同但 b23 真值相同)
        first = list(best_of.values())[0]
        data_v, _ = R.load_vw_data(first[1]["var"], add_yw=True)
        c = data_v[ir]
        yh = c["yh"]
        ax.plot(yh, c["yfit"], "k-", lw=2.4, label="DNS truth")
        for eng, (name, g) in best_of.items():
            ax.plot(yh, g["preds"][ir], lss[eng], lw=1.9, color=colors[eng],
                    label=f"{name} (NMSE={g['nmse'][ir]:.3f})")
        ax.set_xlabel(r"$y/h$")
        ax.set_title(rf"$Ro_\tau$={RO_LIST[ir]:g}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$b_{23}=\langle vw\rangle / u_\tau^2$")
    axes[0].legend(fontsize=9)
    fig.suptitle("b23 profile: truth vs best group per method", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, "fig4_profiles.png"), dpi=150)
    plt.close(fig)

# ---------- fig 5: LGF training history ----------
hist_fig_available = False
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=False)
for ai, var in enumerate(VARS):
    for row, eng in enumerate(["lgf_go", "lgf_mg"]):
        ax = axes[row][ai]
        hj = os.path.join(BASE, f"runs_vwonly_{DIR_TAG[eng]}_{eng}", var,
                          f"vwonly_{var}_{eng}.jsonl")
        if not os.path.exists(hj):
            continue
        gens, losses, secs = [], [], []
        with open(hj, encoding="utf-8") as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                    gens.append(d["gen"]); losses.append(d["best_loss"])
                    secs.append(d.get("seconds", d["gen"]))
                except Exception:
                    pass
        if not gens:
            continue
        hist_fig_available = True
        ax.plot(secs, losses, "-", lw=1.6, color=colors[eng])
        ax.scatter([secs[-1]], [losses[-1]], s=30, color=colors[eng], zorder=3)
        ax.set_yscale("log")
        ax.set_title(f"{eng}/{var}", fontsize=10)
        ax.grid(alpha=0.3, ls=":")
        if row == 1:
            ax.set_xlabel("elapsed (s)")
        if ai == 0:
            ax.set_ylabel("best loss")
if hist_fig_available:
    fig.suptitle("LGF training history: best loss vs elapsed time (V4, GLM-5.3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, "fig5_history.png"), dpi=150)
plt.close(fig)

# ---------- fig 6: NMSE heatmap ----------
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
fig.savefig(os.path.join(FIGD, "fig6_nmse_heat.png"), dpi=150)
plt.close(fig)

print("\nfigures written to", FIGD)
for f in sorted(os.listdir(FIGD)):
    print(" ", f, os.path.getsize(os.path.join(FIGD, f)) // 1024, "KB")
