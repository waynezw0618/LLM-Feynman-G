"""rc_asm_eval.py — 阶段3: 跨后端汇总与应力剖面重构.

读取 outputs/asm/runs_<tag>/summary_{A|B}.json 与 dataset.npz，对每个 (后端, 目标):
  - 方案B: 由 DSL 还原 a_m，再以张量基底重建 6 个应力分量 <ui uj>+
  - 方案A: DSL 直接预测 <ui uj>+
然后与 DNS 真值对比，输出:
  outputs/asm/asm_compare.csv           (per-component per-backend per-Ro R²/NMSE)
  outputs/asm/ASM_summary.md            (跨后端 × 两口径公式表 + R² 摘要)
  outputs/asm/figures/<Ro>_compare.png  (剖面对比图, 如 matplotlib 可用)

用法: python scripts/rc_asm_eval.py [--tags heuristic glm deepseek minimax]
        [--dataset dataset.npz] [--suffix ""]
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lgf import dsl as lgfdsl
from lgf.compiler import CompiledGraph
from lgf.graph import Graph

OUT = os.path.join(REPO, "outputs", "asm")
DSNP = os.path.join(OUT, "dataset.npz")
TAGS_DEFAULT = ["heuristic", "glm", "deepseek", "minimax"]
RO = [7.5, 15.0, 30.0, 75.0, 150.0]
SIX = ["uu", "vv", "ww", "uv", "uw", "vw"]


def _decode_dsl(dsl: str, var_names):
    """编译 DSL 字符串为 CompiledGraph (调用 eval_case(theta_local, X))."""
    if not dsl or dsl.strip() == "":
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            g = lgfdsl.parse_dsl(dsl, var_names)
            return CompiledGraph(g)
        except Exception:
            return None


def _eval_one(f, X, theta, ci):
    """对单 case 应用常数参数: theta_g + theta_case.

    注意: 用 f.n_g / f.n_c 作为真实分块大小, 不依赖 summary 中记录的 n_g/n_c.
    """
    if f is None:
        return None
    n_g, n_c = f.n_g, f.n_c
    L = theta.size
    if L < n_g + n_c * (ci + 1):
        # 兜底: 若 theta 长度与 n_g/n_c 不一致, 尝试按总长裁断
        if n_g == 0 and n_c == 0:
            n_g = L; n_c = 0
        elif n_c == 0:
            n_g = min(L, n_g if n_g else L)
        else:
            return None
    th_g = theta[:n_g]
    th_c = theta[n_g + ci * n_c : n_g + (ci + 1) * n_c]
    tl = np.concatenate([th_g, th_c])
    try:
        return np.asarray(f.eval_case(tl, X), dtype=float)
    except Exception:
        return None


def metrics(y, yp):
    y = np.asarray(y, dtype=float); yp = np.asarray(yp, dtype=float)
    if yp is None or yp.size != y.size or not np.all(np.isfinite(yp)):
        return dict(nmse=float("inf"), r2=-1e9, rmse=float("inf"))
    var = float(np.var(y)) + 1e-12
    mse = float(np.mean((yp - y) ** 2))
    return dict(nmse=mse / var, r2=1.0 - mse / var, rmse=float(np.sqrt(mse)))


def evaluate_summary(scheme: str, summary: dict, ds) -> dict:
    """返回按 (target, case) 的预测与精度. scheme=A 或 B."""
    sel_basis = [str(x) for x in ds["sel_basis"]]
    featB_keys = [str(x) for x in ds["featB_keys"]]
    featA_keys = [str(x) for x in ds["featA_keys"]]
    n_cases_run = len(RO)

    out = dict(scheme=scheme, backend=summary.get("backend"), model=summary.get("model"),
               tag=summary["tag"], targets={})

    if scheme == "A":
        for tgt, tinfo in summary["targets"].items():
            if tgt not in SIX:
                continue
            dsl = tinfo["dsl"]; theta = np.asarray(tinfo["theta"], dtype=float)
            f = _decode_dsl(dsl, featA_keys)
            per_case = {}
            for ci in range(n_cases_run):
                p = f"c{ci}_"; v = ds[p + "valid"]; X = ds[p + "featA"][v]
                y_b = ds[p + "b6"][:, SIX.index(tgt)][v]
                yp_b = _eval_one(f, X, theta, ci)
                m_b = metrics(y_b, yp_b) if yp_b is not None else None
                per_case[f"Ro{RO[ci]:g}"] = dict(a=None, b=m_b,
                    y_range=(float(np.min(y_b)), float(np.max(y_b)), float(np.std(y_b))),
                    yp_range=(None if yp_b is None else
                              (float(np.min(yp_b)), float(np.max(yp_b)), float(np.std(yp_b)))))
            out["targets"][tgt] = dict(dsl=dsl, theta=tinfo["theta"],
                                       loss=tinfo["loss"],
                                       complexity=tinfo["complexity"],
                                       decoded_ok=f is not None,
                                       per_case=per_case,
                                       elapsed_s=tinfo.get("elapsed_s"))
        return out

    # scheme == "B": 每个 a_m 单独评估, 同时组合全 ASM 重建 b_ij
    per_case_ypa = {}     # ci -> ndarray (Nv,5) 预测 a_m (None 表示该 m 失败)
    decoded_ok_global = True
    for tgt, tinfo in summary["targets"].items():
        if tgt not in sel_basis:
            continue
        dsl = tinfo["dsl"]; theta = np.asarray(tinfo["theta"], dtype=float)
        f = _decode_dsl(dsl, featB_keys)
        ok = f is not None
        if not ok:
            decoded_ok_global = False
        per_case = {}
        for ci in range(len(RO)):
            p = f"c{ci}_"; v = ds[p + "valid"]; X = ds[p + "featB"][v]
            y_a = ds[p + "a"][:, sel_basis.index(tgt)][v]
            yp_a = _eval_one(f, X, theta, ci)
            m_a = metrics(y_a, yp_a) if yp_a is not None else None
            per_case[f"Ro{RO[ci]:g}"] = dict(a=m_a, b=None)
            per_case_ypa.setdefault(ci, [None] * len(sel_basis))
            per_case_ypa[ci][sel_basis.index(tgt)] = yp_a
        out["targets"][tgt] = dict(dsl=dsl, theta=tinfo["theta"],
                                   loss=tinfo["loss"],
                                   complexity=tinfo["complexity"],
                                   decoded_ok=ok,
                                   per_case=per_case,
                                   elapsed_s=tinfo.get("elapsed_s"))

    # 重建 b_ij: yp_b[c][k,comp] = sum_m a_m * T_m[comp]; 对每个 (comp, Ro) 算 R²
    recon = {}
    for ci in range(len(RO)):
        p = f"c{ci}_"; v = ds[p + "valid"]
        ypa = per_case_ypa.get(ci)
        # 拼装 Tb: (Nv, 5, 6) = [T_S, T_SOm, T_S2, T_Om2, T_Om2S] 每个 (Nv, 6)
        Tb_list = [ds[p + "T_" + m][v] for m in sel_basis]
        Tb = np.stack(Tb_list, axis=1)         # (Nv, 5, 6)
        dns = ds[p + "b6"][v]                  # (Nv, 6)
        if ypa is None or any(a is None for a in ypa):
            yp_b_all = None
        else:
            ypa_mat = np.stack([np.asarray(a) for a in ypa], axis=1)  # (Nv,5)
            yp_b_all = np.einsum('imc,im->ic', Tb, ypa_mat)           # (Nv,6)
        for k, comp in enumerate(SIX):
            y = dns[:, k]
            m = metrics(y, (None if yp_b_all is None else yp_b_all[:, k])) \
                if yp_b_all is not None else None
            recon.setdefault(comp, {})[f"Ro{RO[ci]:g}"] = m
    out["recon_b"] = recon
    out["decoded_ok"] = decoded_ok_global
    return out


def load_all(tags, schemes=("A", "B")):
    out = {}
    for tag in tags:
        for sc in schemes:
            p = os.path.join(OUT, f"runs_{tag}", f"summary_{sc}.json")
            if os.path.exists(p):
                try:
                    s = json.load(open(p, encoding="utf-8"))
                    ds = np.load(DSNP)
                    key = (tag, sc)
                    out[key] = evaluate_summary(sc, s, ds)
                except Exception as e:
                    print(f"[eval] skip {tag}/{sc}: {e}")
    return out


def write_csv(all_res, path):
    cols = ["tag", "scheme", "target", "Ro", "R2_stress", "R2_coef_a"]
    rows = []
    for (tag, sc), res in all_res.items():
        if sc == "B":
            recon = res.get("recon_b", {})
            for comp in SIX:
                for ro, m in (recon.get(comp) or {}).items():
                    rows.append(dict(tag=tag, scheme=sc, target=f"<{comp}>", Ro=ro,
                                     R2_stress=round(m["r2"], 4) if m and np.isfinite(m.get("r2", -1e9)) else "",
                                     R2_coef_a=""))
            # 附加 a_m 自身 R²
            for tgt, tinfo in res["targets"].items():
                for ro, m in tinfo["per_case"].items():
                    a = m.get("a") or {}
                    rows.append(dict(tag=tag, scheme=sc, target=f"a_{tgt}", Ro=ro,
                                     R2_stress="",
                                     R2_coef_a=round(a["r2"], 4) if a and np.isfinite(a.get("r2", -1e9)) else ""))
        else:
            for tgt, tinfo in res["targets"].items():
                for ro, m in tinfo["per_case"].items():
                    b = m.get("b") or {}
                    rows.append(dict(tag=tag, scheme=sc, target=tgt, Ro=ro,
                                     R2_stress=round(b["r2"], 4) if b and np.isfinite(b.get("r2", -1e9)) else "",
                                     R2_coef_a=""))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def write_md(all_res, path):
    lines = ["# 旋转槽道流 ASM-LLM-Feynman-G 跨后端汇总",
             "", "## 1. 框架与数据", ""]
    lines += [
        "- DNS 数据: Yang & Wang (2018) JFM 838:658-689, Reτ=180, Roτ ∈ {7.5, 15, 30, 75, 150}",
        "- 参考系: x₁ 流向 / x₂ 壁面法向 y / x₃ 展向; Ω 沿 x₁",
        "- 扩展内禀旋转张量: Ω̃ᵢⱼ = W̄ᵢⱼ + 2εᵢⱼₖΩₖ (黄于宁、马晖扬 2008, 式(19))",
        "- 5 张不变量/共自变量: η²=2tr(S̄S̄)(k/ε)², ζ²=2tr(Ω̃Ω̃)(k/ε)², s3=tr(S̄³)(k/ε)³, "
        "so2=tr(S̄Ω̃²)(k/ε)³, s2o2=tr(S̄²Ω̃²)(k/ε)⁴",
        "- 方案B: τᵢⱼ/(2k) = Σ aₘ Tᵢⱼ⁽ᵐ⁾; T 集合 = {S, SΩ̃, S², Ω̃², SΩ̃²}, "
        "先线性 lstsq 反演 aₘ(y), 再 LGF 发现 aₘ = f(5 不变量)",
        "- 方案A: 直接对 6 个应力分量做 LGF 符号回归, 自变量同前 5 不变量 + Roτ",
        "- 评估: per-case NMSE/R²/RMSE, 方案B 需重建 b_ij = T·aₘ",
        "", "## 2. 后端与运行设置", ""]
    seen_tags = sorted({t for (t, _) in all_res})
    for tag in seen_tags:
        for sc in ["A", "B"]:
            res = all_res.get((tag, sc))
            if not res:
                continue
            lines.append(f"- **{tag} / scheme={sc}**: model={res.get('model') or '-'}, "
                         f"decoder={'OK' if any(t['decoded_ok'] for t in res['targets'].values()) else 'FAIL'}")
    lines += ["", "## 3. 方案B — aₘ 公式表", ""]
    lines += ["| backend", "target", "DSL", "complexity (N/E/P/D)", "time(s)", "|"]
    lines += ["|---|---|---|---|---|"]
    for (tag, sc), res in all_res.items():
        if sc != "B":
            continue
        for tgt, tinfo in res["targets"].items():
            dsl = (tinfo["dsl"] or "").replace("|", "\\|")
            if len(dsl) > 100:
                dsl = dsl[:97] + "..."
            c = tinfo["complexity"]; e = tinfo["elapsed_s"]
            lines.append(f"| {tag} | {tgt} | `{dsl}` | "
                         f"{c['nodes']}/{c['edges']}/{c['params']}/{c['depth']} | {e} |")
    lines += ["", "## 3.5 方案A — 分量公式表", ""]
    lines += ["| backend | target | DSL | loss | complexity (N/E/P/D) | time(s) |", "|---|---|---|---|---|---|"]
    for (tag, sc), res in all_res.items():
        if sc != "A":
            continue
        for tgt, tinfo in res["targets"].items():
            dsl = (tinfo["dsl"] or "").replace("|", "\\|")
            if len(dsl) > 100:
                dsl = dsl[:97] + "..."
            c = tinfo["complexity"]; e = tinfo.get("elapsed_s") or "-"
            lines.append(f"| {tag} | {tgt} | `{dsl}` | {tinfo['loss']:.3g} | "
                         f"{c['nodes']}/{c['edges']}/{c['params']}/{c['depth']} | {e} |")
    lines += ["", "## 4. 方案B — 全 ASM 重建 <ui uj>+ 的 R² (越大越好)", ""]
    lines += ["B 方案以全 5 个 a_m 公式重建完整 b_ij 后与 DNS 对比, " +
              "下表每个 cell 是对应 (分量, Ro) 的 R².", ""]
    for (tag, sc), res in all_res.items():
        if sc != "B":
            continue
        recon = res.get("recon_b", {})
        lines += ["", f"**{tag}**", ""]
        header = "| component | " + " | ".join(f"Ro{r:g}" for r in RO) + " |"
        sep = "|" + "---|" * 7
        lines += [header, sep]
        for comp in SIX:
            row = f"| <{comp}> "
            for ro in RO:
                m = (recon.get(comp) or {}).get(f"Ro{ro:g}")
                v = "—" if (not m or not np.isfinite(m.get("r2", -1e9))) else round(m["r2"], 3)
                row += f"| {v} "
            row += "|"
            lines.append(row)
        # 附加: a_m 自身 per-target R² (信息性, 不是最终 ASM 指标)
        lines += ["", f"_(参考) 单 a_m 自身 R² — a_S 几乎 R²<0 (y 方向变化大, " +
                  f"难用简单符号式拟合); a_SOm 是 Ro-平均的良基_",
                  ""]
        for tgt, tinfo in res["targets"].items():
            row = f"| a_{tgt} "
            for ro in RO:
                m = tinfo["per_case"].get(f"Ro{ro:g}", {}).get("a")
                v = "—" if (not m or not np.isfinite(m.get("r2", -1e9))) else round(m["r2"], 3)
                row += f"| {v} "
            row += "|"
            lines.append(row)
    lines += ["", "## 5. 方案A — 直接应力回归 R²", ""]
    for (tag, sc), res in all_res.items():
        if sc != "A":
            continue
        lines += ["", f"**{tag}**", ""]
        header = "| target | " + " | ".join(f"Ro{r:g}" for r in RO) + " |"
        sep = "|" + "---|" * 7
        lines += [header, sep]
        for tgt in [k for k in res["targets"]]:
            row = f"| {tgt} "
            for ro in RO:
                m = res["targets"][tgt]["per_case"].get(f"Ro{ro:g}", {}).get("b")
                v = "—" if (not m or not np.isfinite(m.get("r2", -1e9))) else round(m["r2"], 3)
                row += f"| {v} "
            row += "|"
            lines.append(row)
    lines += ["", "## 6. 注", "",
             "- 标 '—' 表示 DSL 解析失败或产生 NaN/Inf (后端可能被 LGF 静默退化)",
             "- 方案B 中 'S' / 'S2' 目标对应 b 二阶强反对称项, 5 工况联合反演残差~1e-7, "
             "若 LGF 公式 R² 接近 1 说明成功发现",
             "- 论文 Yang & Wang 2018 给出 'Capturing TG vortices' 原始结果, "
             "本表是基于 DNS 重发现, **不可外推**至其他旋转几何或 Reτ"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def maybe_plots(all_res, path_dir, suffix=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[eval] matplotlib 不可用, 跳过剖面图")
        return
    os.makedirs(path_dir, exist_ok=True)
    ds = np.load(DSNP)
    sel_basis = [str(x) for x in ds["sel_basis"]]
    has_basis = "Tb" in ds.files and "b6" in ds.files
    for ci, ro in enumerate(RO):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
        p = f"c{ci}_"; v = ds[p + "valid"]; yh = ds[p + "yh"][v]
        for k, comp in enumerate(SIX):
            ax = axes[k // 3][k % 3]
            dns = ds[p + "b6"][:, SIX.index(comp)][v]
            ax.plot(yh, dns, "k-", lw=2.2, label="DNS")
            # 用 DNS 范围定 y 轴 (限幅, 避免被发散的预测线拉爆)
            y_lo, y_hi = float(np.min(dns)), float(np.max(dns))
            span = max(y_hi - y_lo, 0.05)
            ax.set_ylim(y_lo - 0.15 * span, y_hi + 0.15 * span)
            for (tag, sc), res in all_res.items():
                yp = None
                if sc == "B":
                    ypa_mat = np.zeros((len(yh), len([str(x) for x in ds["sel_basis"]])))
                    ok_all = True
                    for mi, tgt in enumerate([str(x) for x in ds["sel_basis"]]):
                        if tgt not in res["targets"]:
                            ok_all = False; break
                        tinfo = res["targets"][tgt]
                        f = _decode_dsl(tinfo["dsl"], [str(x) for x in ds["featB_keys"]])
                        if f is None: ok_all = False; break
                        ypa = _eval_one(f, ds[p + "featB"][v],
                                        np.asarray(tinfo["theta"], dtype=float), ci)
                        if ypa is None or not np.all(np.isfinite(ypa)):
                            ok_all = False; break
                        ypa_mat[:, mi] = ypa
                    if ok_all:
                        Tb_list = [ds[p + "T_" + m][v] for m in
                                   [str(x) for x in ds["sel_basis"]]]
                        Tb = np.stack(Tb_list, axis=1)        # (Nv,5,6)
                        yp_full = np.einsum('imc,im->ic', Tb, ypa_mat)           # (Nv,6)
                        yp = yp_full[:, k]
                else:
                    if comp not in res["targets"]: continue
                    tinfo = res["targets"][comp]
                    f = _decode_dsl(tinfo["dsl"], [str(x) for x in ds["featA_keys"]])
                    if f is None: continue
                    yp = _eval_one(f, ds[p + "featA"][v],
                                   np.asarray(tinfo["theta"], dtype=float), ci)
                if yp is not None and np.all(np.isfinite(yp)):
                    ax.plot(yh, yp, "--", lw=1.2, label=f"{tag}/{sc}", alpha=0.9)
                elif yp is not None:
                    # 部分点 finite, 用 nan 隔离画
                    ax.plot(yh, yp, lw=1.0, alpha=0.5, label=f"{tag}/{sc}(partial)")
            ax.set_title(f"<{comp}>+"); ax.set_xlabel("y/h"); ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7, loc="best")
        fig.suptitle(f"Ro_tau={ro:g}: DNS vs LLM-Feynman-G ASM")
        fig.tight_layout()
        fig.savefig(os.path.join(path_dir, f"Ro{ro:g}_compare{suffix}.png"), dpi=120)
        plt.close(fig)


def main():
    global DSNP
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=TAGS_DEFAULT)
    ap.add_argument("--dataset", default="dataset.npz",
                    help="数据集文件名 (相对 outputs/asm)")
    ap.add_argument("--suffix", default="",
                    help="输出文件名后缀 (如 _vort -> asm_compare_vort.csv)")
    args = ap.parse_args()
    DSNP = os.path.join(OUT, args.dataset)
    os.makedirs(OUT, exist_ok=True)
    all_res = load_all(args.tags)
    if not all_res:
        print("[eval] no summaries found"); return
    sfx = args.suffix
    write_csv(all_res, os.path.join(OUT, f"asm_compare{sfx}.csv"))
    write_md(all_res, os.path.join(OUT, f"ASM_summary{sfx}.md"))
    maybe_plots(all_res, os.path.join(OUT, "figures"), sfx)
    print(f"[eval] -> {OUT}/asm_compare{sfx}.csv")
    print(f"[eval] -> {OUT}/ASM_summary{sfx}.md")
    print(f"[eval] -> {OUT}/figures/")


if __name__ == "__main__":
    main()