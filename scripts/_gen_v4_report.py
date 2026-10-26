# -*- coding: utf-8 -*-
"""_gen_v4_report.py — V4/V4b 轮 (GLM-5.3 + L1 五项 + clone 修复) 结果报告生成器.

读 outputs/asm/runs_vwonly_{v4_pysr, v4b_lgf_go, v4b_lgf_mg}/{v,g,e,m}/result.json,
按正式口径 (NMSE 分母 = var_c) 重算 12 组指标, 生成:
  outputs/asm/ASM_vwonly_v4_report.md

口径:
  - PySR: r["best"]["sympy"]; per_case nmse 来自 r["best"]["metrics"]
  - LGF : entries 中 loss_paramfit 最小者; preds 用 rc_asm_vw_only.lgf_predict_*
  - NMSE 分母统一用 case["var_c"]  (与 figs 脚本一致)
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, r"d:\AI\LLM-Feynman-G\scripts")
import rc_asm_vw_only as R

BASE = r"d:\AI\LLM-Feynman-G\outputs\asm"
OUT = os.path.join(BASE, "ASM_vwonly_v4_report.md")

ENGINES = ["pysr", "lgf_go", "lgf_mg"]
VARS = ["v", "g", "e", "m"]
RO_LIST = [7.5, 15.0, 30.0, 75.0, 150.0]
DIR_TAG = {"pysr": "v4", "lgf_go": "v4b", "lgf_mg": "v4b"}
ENG_LABEL = {"pysr": "PySR", "lgf_go": "LGF-g (ginit-only)",
             "lgf_mg": "LGF-m (full)"}
VAR_LABEL = {"v": "v (W̄)", "g": "g (W̄+Ω)", "e": "e (W̄+2Ω)", "m": "m (mix)"}


def official(preds, data):
    per_nmse, per_rho = [], []
    for p, c in zip(preds, data):
        y = c["yfit"]
        per_nmse.append(float(np.mean((p - y) ** 2) / c["var_c"]))
        per_rho.append(float(np.corrcoef(p, y)[0, 1]))
    pooled = float(np.corrcoef(np.concatenate(preds),
                              np.concatenate([c["yfit"] for c in data]))[0, 1])
    return per_nmse, per_rho, pooled


def subst_theta(dsl, theta):
    """把 DSL 里的 C#/K# 替换成数值, 便于阅读.
    保名修复后 params() 按名排序, theta 顺序 = sorted(C# names)."""
    names = sorted(set(re.findall(r"\bC\d+\b", dsl)), key=lambda s: int(s[1:]))
    out = dsl
    for n, t in zip(names, theta or []):
        out = re.sub(rf"\b{n}\b", f"({t:.5g})", out)
    return out


def load_groups():
    G = {}
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
                n_par = b.get("complexity")
                theta = None
                rec = b["metrics"]["loss_vw"]
            else:
                b = min(r["entries"], key=lambda x: x["loss_paramfit"])
                if var == "m":
                    preds, _ = R.lgf_predict_mix(b["dsl"], b["theta"], data, vn)
                else:
                    preds = R.lgf_predict_vw(b["dsl"], b["theta"], data, vn)
                formula = subst_theta(b["dsl"], b["theta"])
                n_par = len(set(re.findall(r"\bC\d+\b", b["dsl"])))
                theta = b["theta"]
                rec = b["metrics"]["loss_vw"]
            nmse, rho, pool = official(preds, data)
            G[f"{eng}/{var}"] = dict(
                eng=eng, var=var, dsl_raw=(b["dsl"] if eng != "pysr" else b["sympy"]),
                formula=formula, theta=theta, n_par=n_par, rec=rec,
                nmse=nmse, rho=rho, pool=pool, worst=max(nmse),
                nmse_mean=float(np.mean(nmse)), rho_mean=float(np.nanmean(rho)),
                elapsed=r.get("elapsed_s"),
                n_pareto=(len(r.get("pareto", [])) if eng == "pysr" else None),
                llm_stats=r.get("llm_stats"))
    return G


def struct_tags(formula):
    f = formula.lower()
    tags = []
    for kw, lb in [("log(", "log"), ("exp(", "exp"), ("tanh(", "tanh"),
                   ("sqrt(", "sqrt"), ("roloc", "roloc"), ("yw", "yw"),
                   ("ret", "ret"), ("zeta2", "(ω̃²/2)"), ("zeta", "(ω̃)")]:
        if kw in f:
            tags.append(lb)
    return ", ".join(dict.fromkeys(tags)) if tags else "-"


def main():
    G = load_groups()
    rank = sorted(G.items(), key=lambda kv: kv[1]["nmse_mean"])
    med = np.median([g["nmse_mean"] for _, g in rank]) if rank else 1.0

    L = []
    A = L.append
    A("# vw-only 闭合建模 V4 轮报告：LGF（GLM-5.3）vs PySR")
    A("")
    A("> 口径：本轮为 **b23 系数函数反演**（`<vw>` 单分量），非真 ASM 方程求解；")
    A("> 特征池 18 维（e/m 变体 44 维）+ 真壁距 `yw`；NMSE 分母统一 = `var_c`（逐工况方差）。")
    A("")
    A("## 0. 本轮改了什么（L1 五项修复 + 后端升级 + V4b 缓存修复）")
    A("")
    A("| # | 项 | 内容 | 状态 |")
    A("|---|----|------|------|")
    A("| L1-1 | DSL/参数对齐 | `walk()` 改标准 preorder；`_normalize_params` 保名；`params()` 按名排序 | ✅ |")
    A("| L1-2 | paramfit 限时 | deadline callback + `screen_budget_s`/`gp_fit_budget_s` 双旋钮 + V4 重构（固定子采样/初筛/存活者缓存/final refit） | ✅ |")
    A("| L1-3 | 解析率 | 后端换 GLM-5.3；`thinking=enabled`+`reasoning_effort=low`+`max_tokens=16384` | ✅ |")
    A("| L1-4 | 特征标准化 | `yw` 由 `(1-|yh|)·180 ∈[0,180]` → `1-|yh| ∈[0,1]` 无量纲 | ✅ |")
    A("| L1-5 | 多 start | 对数尺度 start（±10^U(-3,3)）+ final refit 2× start / 2× maxiter | ✅ |")
    A("| V4b | GP 缓存修复 | `Graph.clone()` 拷贝父代 loss → 子代被误判存活跳过拟合 → 子代入种群前强制 `loss=inf` 重置 | ✅ |")
    A("| V4b | G_INIT 提示 | 补 `log(yw)` 对数律家族（上轮 SOTA 公式均含 log(yw)） | ✅ |")
    A("")
    A(f"## 1. 总排名（{len(G)}/12 组，按 5 工况 NMSE 均值升序）")
    A("")
    if not rank:
        A("_（暂无结果）_")
    else:
        A("| 名次 | 组 | 方法 | 目标量 | NMSE均值 | 最差工况 | ρ均值 | ρ(pooled) | 记录loss_vw | 参数量 | 耗时(s) |")
        A("|----|----|------|--------|--------:|--------:|------:|----------:|------------:|-------:|-------:|")
        for i, (name, g) in enumerate(rank):
            A(f"| {i+1} | `{name}` | {ENG_LABEL[g['eng']]} | {VAR_LABEL[g['var']]} | "
              f"{g['nmse_mean']:.4f} | {g['worst']:.3f} | {g['rho_mean']:.4f} | "
              f"{g['pool']:.4f} | {g['rec']:.4f} | {g['n_par']} | {g['elapsed']:.0f} |")
    A("")
    if rank:
        winner = rank[0]
        A(f"**排名第一**：`{winner[0]}`（{ENG_LABEL[winner[1]['eng']]} / {VAR_LABEL[winner[1]['var']]}），"
          f"NMSE均值 = {winner[1]['nmse_mean']:.4f}，最差工况 = {winner[1]['worst']:.3f}。")
    A("")
    A("## 2. 逐工况 NMSE（正式口径 = var_c）")
    A("")
    if rank:
        A("| 组 | " + " | ".join(f"Ro{r:g}" for r in RO_LIST) + " | 均值 |")
        A("|----|" + "|".join(["----:"] * (len(RO_LIST) + 1)) + "|")
        for name, g in rank:
            A(f"| `{name}` | " + " | ".join(f"{v:.3f}" for v in g["nmse"]) +
              f" | {g['nmse_mean']:.3f} |")
    A("")
    A("## 3. 各组最优公式")
    A("")
    A("### 3.1 LGF（GLM-5.3 后端）— 8 组")
    A("")
    for eng in ["lgf_mg", "lgf_go"]:
        for var in VARS:
            key = f"{eng}/{var}"
            if key not in G:
                continue
            g = G[key]
            A(f"**`{key}`**（NMSE均值={g['nmse_mean']:.4f}，{g['n_par']} 参数）")
            A("")
            A("```")
            A(g["formula"])
            A("```")
            A("")
    A("### 3.2 PySR — 4 组")
    A("")
    for var in VARS:
        key = f"pysr/{var}"
        if key not in G:
            continue
        g = G[key]
        A(f"**`{key}`**（NMSE均值={g['nmse_mean']:.4f}，复杂度 {g['n_par']}）")
        A("")
        A("```")
        A(g["formula"])
        A("```")
        A("")
    A("## 4. 图")
    A("")
    for fn, cap in [("fig1_rank_loss_rho.png", "NMSE vs 形状相关系数（log x）"),
                    ("fig2_bar_per_ro.png", "各组 × 5 工况 NMSE 分组柱状（log y）"),
                    ("fig3_rho_per_ro.png", "Top-5 组 ρ 随 Roτ 曲线"),
                    ("fig4_profiles.png", "每方法最优组 b23 剖面（Ro=7.5/30/150）"),
                    ("fig5_history.png", "LGF 8 组训练历史（best loss vs 时间）"),
                    ("fig6_nmse_heat.png", "12 组 × 5 工况 NMSE 热图")]:
        A(f"![{cap}](figs_vwonly_v4/{fn})")
        A("")
        A(f"*图：{cap}*")
        A("")
    A("## 5. 物理解释")
    A("")
    A("_（见下文，由本轮获胜公式结构给出）_")
    A("")
    A("## 6. 局限与口径说明")
    A("")
    A("- 所有跑批为系数函数拟合，非 ASM 方程求解。")
    A("- `yw` 改无量纲 [0,1] 后，PySR 的 `1/yw`、`log(yw)` 动态范围变化，需 V4c 尺度对照确认结论稳健。")
    A("")

    os.makedirs(BASE, exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"[report] {len(G)}/12 groups -> {OUT}")
    for i, (n, g) in enumerate(rank):
        print(f"  {i+1:>2}. {n:<10} nmse={g['nmse_mean']:.4f} worst={g['worst']:.3f} "
              f"rho={g['rho_mean']:.4f} rec={g['rec']:.4f} {g['n_par']}par")


if __name__ == "__main__":
    main()
