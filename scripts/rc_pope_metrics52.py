"""rc_pope_metrics52.py — Weatheritt & Sandberg 2016 (JCP 325:22-37) §5.2 评估口径.

严格对齐论文式 (21)-(24), 适配到本问题 (槽道湍流 DNS 剖面, 无 a posteriori RANS):
  rho_a(y)    式(21): 逐点模型-DNS 各向异性张量 b_ij 余弦对齐
                      (论文的 a posteriori 对齐在 RANS 解上算; 我们在 DNS 平均场
                      上算, 属 on-design a priori 对齐 —— 诚实声明此差异)
  rho_a,case  式(22): ∫_y rho_a dy, 槽道 y/h∈[-1,1] (另报均值 = 积分/2)
  tau_r       式(23): 法向应力比 tau22/tau11 = (b22+1/3)/(b11+1/3) 剖面相对误差
  tau_e       式(24): ∫ ||tau_model - tau_ref||_F dy / |tau_ref_kk|;
                      实现: mean_y( ||dtau||_F / (2k) ) (点相对 Frobenius 误差的
                      剖面均值, tau_kk=2k 逐点; 论文分母口径声明于此)
  偏相关      式(25)-(28): 控 tau_e 后 Fit 与 rho_a,case 的 residual-residual 相关.
                      样本 = (arm, case): 4 arm x 5 case = 20 点
                      (论文 50 个独立模型; 我们的 arm 非独立样本, 解释需谨慎)

用法:
  python scripts/rc_pope_metrics52.py --pairs \
      heuristic_pope_wy=dataset_pope_wy.npz heuristic_pope3_wy=dataset_pope3_wy.npz \
      heuristic_pope_wy_rt=dataset_pope_wy_rt.npz heuristic_pope_gsrt=dataset_pope_gsrt.npz \
      --out ASM_metrics52_gsrt.md
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_eval import _decode_dsl, _eval_one  # noqa: E402

OUT = os.path.join(REPO, "outputs", "asm")
FIG = os.path.join(OUT, "figures")
RO = [7.5, 15.0, 30.0, 75.0, 150.0]
SIX = ["uu", "vv", "ww", "uv", "uw", "vw"]


def frob6(v):
    """6 分量存储 [uu,vv,ww,uv,uw,vw] 的对称张量 Frobenius 范数."""
    return np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2 + v[..., 2] ** 2
                   + 2.0 * (v[..., 3] ** 2 + v[..., 4] ** 2 + v[..., 5] ** 2))


def reconstruct(tag, ds):
    """解码 5 目标公式 → 逐 case 重建 b6_model (全 valid 点). 返回 per-case dict."""
    summary = json.load(open(os.path.join(OUT, f"runs_{tag}", "summary_B.json"),
                             encoding="utf-8"))
    sel_basis = [str(x) for x in ds["sel_basis"]]
    feat_keys = [str(x) for x in ds["featB_keys"]]
    per_case = {}
    for ci in range(len(RO)):
        p = f"c{ci}_"; v = ds[p + "valid"]
        ypa = []
        ok = True
        for tgt in sel_basis:
            tinfo = summary["targets"][tgt]
            f = _decode_dsl(tinfo["dsl"], feat_keys)
            if f is None:
                ok = False
                break
            yp = _eval_one(f, ds[p + "featB"][v], np.asarray(tinfo["theta"]), ci)
            ypa.append(yp)
        if not ok:
            per_case[ci] = None
            continue
        Tb = np.stack([ds[p + "T_" + m][v] for m in sel_basis], axis=1)
        ypa_mat = np.stack(ypa, axis=1)
        rec = np.einsum("imc,im->ic", Tb, ypa_mat)
        per_case[ci] = dict(rec=rec, dns=ds[p + "b6"][v], yh=ds[p + "yh"][v],
                            k=ds[p + "k"][v], valid=v)
    return per_case, summary, sel_basis


def case_metrics(pc):
    """式(21)-(24) 逐 case 指标."""
    rec, dns, yh, k = pc["rec"], pc["dns"], pc["yh"], pc["k"]
    # (21) rho_a(y): 6 分量余弦 (对称张量内积 <A,B>_F)
    num = np.sum(rec * dns, axis=1)
    den = frob6(rec) * frob6(dns)
    rho = np.where(den > 1e-300, num / np.maximum(den, 1e-300), np.nan)
    # (22) rho_a,case: y/h∈[-1,1] 积分 (梯形); 均值 = 积分/2
    o = np.argsort(yh)
    rho_int = float(np.trapezoid(rho[o], yh[o])) if np.isfinite(rho).all() else \
        float(np.trapezoid(np.nan_to_num(rho[o]), yh[o]))
    # (23) tau_r = (b22+1/3)/(b11+1/3) 剖面相对误差 (点wise |Δ|/|dns|, 幅值截断)
    tr_m = (rec[:, 1] + 1.0 / 3.0) / np.maximum(np.abs(rec[:, 0] + 1.0 / 3.0), 0.05)
    tr_d = (dns[:, 1] + 1.0 / 3.0) / np.maximum(np.abs(dns[:, 0] + 1.0 / 3.0), 0.05)
    tr_err = float(np.mean(np.abs(tr_m - tr_d)))
    # (24) tau_e: mean_y( ||dtau||_F / (2k) ), dtau = 2k*(b_model - b_dns) 分量式
    dtau_f = 2.0 * k[:, None] * (rec - dns)
    dtau_f[:, 3:] *= np.sqrt(2.0)  # offdiag Frobenius 权重
    tau_e = float(np.mean(frob6(dtau_f) / (2.0 * k)))
    return dict(rho_prof=(yh, rho), rho_int=rho_int, rho_mean=rho_int / 2.0,
                tr_err=tr_err, tau_e=tau_e)


def partial_corr(x, y, z):
    """式(25)-(28): 控 z 后 x-y 偏相关. 返回 r, p, 残差."""
    from scipy import stats

    def resid(a, b):
        sl, ic, _, _, _ = stats.linregress(b, a)
        return a - (sl * b + ic)
    wx = resid(x, z)
    wy = resid(y, z)
    r, p = stats.pearsonr(wx, wy)
    return r, p, wx, wy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", default="ASM_metrics52_gsrt.md")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    arms = {}
    for pair in args.pairs:
        tag, dsname = pair.split("=", 1)
        sp = os.path.join(OUT, f"runs_{tag}", "summary_B.json")
        if not os.path.exists(sp):
            print(f"[m52] 跳过 {tag}: 无 {sp}")
            continue
        ds = np.load(os.path.join(OUT, dsname))
        per_case, summary, sel_basis = reconstruct(tag, ds)
        # arm 级 a priori fitness: 5 目标 stress-weighted loss 均值
        fit = float(np.mean([t["loss"] for t in summary["targets"].values()]))
        arms[tag] = dict(ds=dsname, per_case=per_case, fit=fit, basis=sel_basis,
                         summary=summary)
        print(f"[m52] {tag}: Fit={fit:.4f}, basis={sel_basis}")

    lines = ["# Weatheritt & Sandberg (2016) §5.2 评估 — GS 组合实验", "",
             "口径: 式(21) rho_a(y) 逐点 b_ij 余弦对齐; 式(22) rho_a,case=∫rho_a dy "
             "(y/h∈[-1,1]); 式(23) tau_r=(b22+1/3)/(b11+1/3) 剖面平均绝对误差; "
             "式(24) tau_e=mean_y(||dtau||_F/(2k)).",
             "",
             "⚠ 与论文的差异 (诚实声明): 论文的 rho_a 是 RANS a posteriori 解对 "
             "LES/DNS 的对齐; 我们无 RANS 回代, 在 DNS 平均场上评估, 属 "
             "**on-design a priori 对齐**. 偏相关样本是 4 arm × 5 case = 20 点 "
             "(论文为 50 个独立模型), arm 间非独立抽样, r 的显著性解释需谨慎.", ""]

    # ---- 主表: 每 arm 每 case 的 rho_a,case / tau_r / tau_e ----
    lines += ["## 1. 式(21)(22): 各向异性对齐 rho_a,case (=∫rho_a dy, 满分 2.0)", "",
              "| arm | " + " | ".join(f"Ro{ro:g}" for ro in RO) + " | mean |",
              "|---" * (len(RO) + 2) + "|"]
    tab = {}
    for tag, a in arms.items():
        vals = [case_metrics(a["per_case"][ci])["rho_int"]
                if a["per_case"][ci] else np.nan for ci in range(len(RO))]
        tab[tag] = vals
        lines.append(f"| {tag} | " + " | ".join(f"{v:.3f}" for v in vals)
                     + f" | {np.nanmean(vals):.3f} |")
    lines += ["", "参考: 完美模型 rho_a,case = 2.0 (积分区间长). "
              "线性涡黏 (b=0) 的 rho_a 无定义 (分母为 0), 论文以线性 SST 为基线; "
              "我们的基线是 heuristic_pope_wy (P1a).", ""]

    lines += ["## 2. 式(23): tau_r = tau22/tau11 剖面平均绝对误差 (越小越好)", "",
              "| arm | " + " | ".join(f"Ro{ro:g}" for ro in RO) + " | mean |",
              "|---" * (len(RO) + 2) + "|"]
    for tag, a in arms.items():
        vals = [case_metrics(a["per_case"][ci])["tr_err"]
                if a["per_case"][ci] else np.nan for ci in range(len(RO))]
        lines.append(f"| {tag} | " + " | ".join(f"{v:.3f}" for v in vals)
                     + f" | {np.nanmean(vals):.3f} |")
    lines += ["", "注: tau_r 在 b11+1/3 接近 0 处误差放大 (论文同口径), "
              "分母截断 0.05 防发散.", ""]

    lines += ["## 3. 式(24): tau_e 应力幅值相对误差 (越小越好)", "",
              "| arm | " + " | ".join(f"Ro{ro:g}" for ro in RO) + " | mean |",
              "|---" * (len(RO) + 2) + "|"]
    te_tab = {}
    for tag, a in arms.items():
        vals = [case_metrics(a["per_case"][ci])["tau_e"]
                if a["per_case"][ci] else np.nan for ci in range(len(RO))]
        te_tab[tag] = vals
        lines.append(f"| {tag} | " + " | ".join(f"{v:.4f}" for v in vals)
                     + f" | {np.nanmean(vals):.4f} |")
    lines += ["", "tau_e 是论文中的混淆变量: 单目标优化可能牺牲幅值 "
              "(a_model ≈ C·a_dns, C≠1 而对齐 rho_a 不变), 需在偏相关中控制.", ""]

    # ---- 式(25)-(28): 偏相关 ----
    lines += ["## 4. 式(25)-(28): 控 tau_e 后 Fit ↔ rho_a,case 偏相关", ""]
    X, Y, Z = [], [], []
    for tag, a in arms.items():
        for ci in range(len(RO)):
            if a["per_case"][ci] is None:
                continue
            X.append(a["fit"])
            Y.append(tab[tag][ci])
            Z.append(te_tab[tag][ci])
    X, Y, Z = map(np.array, (X, Y, Z))
    if len(X) >= 5 and np.std(X) > 0 and np.std(Z) > 0:
        r, p, wx, wy = partial_corr(X, Y, Z)
        r0 = np.corrcoef(X, Y)[0, 1]
        lines += [f"- 样本: {len(X)} 个 (arm, case) 点",
                  f"- 简单相关 corr(Fit, rho_a) = {r0:+.3f}",
                  f"- **偏相关 (控 tau_e) r = {r:+.3f}, p = {p:.3g}**",
                  f"- 论文 region I 参照: r = 0.93, p = 1e-11 (50 模型)",
                  "",
                  "逐点明细 (Fit 低=训练好; 论文 Fit 越高越好, 此处 loss 反向):",
                  "",
                  "| arm | Ro | Fit (loss) | rho_a,case | tau_e |",
                  "|---|---|---|---|---|"]
        i = 0
        for tag, a in arms.items():
            for ci in range(len(RO)):
                if a["per_case"][ci] is None:
                    continue
                lines.append(f"| {tag} | {RO[ci]:g} | {X[i]:.4f} | {Y[i]:.3f} "
                             f"| {Z[i]:.4f} |")
                i += 1
    lines.append("")

    # ---- rho_a(y) 剖面图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(RO), figsize=(3.0 * len(RO), 3.6),
                                 sharey=True)
        cols = dict(zip(arms.keys(), ["k", "tab:blue", "tab:orange", "tab:red"]))
        for ci in range(len(RO)):
            ax = axes[ci]
            for tag, a in arms.items():
                pc = a["per_case"][ci]
                if pc is None:
                    continue
                yh, rho = case_metrics(pc)["rho_prof"]
                o = np.argsort(yh)
                ax.plot(rho[o], yh[o], color=cols.get(tag, None),
                        label=tag.replace("heuristic_pope_", ""), lw=1.2)
            ax.set_title(f"$Ro_\\tau$={RO[ci]:g}")
            ax.set_xlabel("$\\rho_a(y)$")
            ax.grid(alpha=0.3)
            ax.set_xlim(-0.2, 1.05)
        axes[0].set_ylabel("$y/h$")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "metrics52_rho_a_profiles.png"), dpi=130)
        plt.close(fig)
        lines += ["图: figures/metrics52_rho_a_profiles.png (5 工况 rho_a(y) 剖面, "
                  "4 arm 对比)", ""]
        print("[m52] figure -> metrics52_rho_a_profiles.png")
    except Exception as e:  # noqa: BLE001
        lines += [f"(图绘制失败: {e})", ""]

    dst = os.path.join(OUT, args.out)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[m52] -> {dst}")


if __name__ == "__main__":
    main()
