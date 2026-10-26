# -*- coding: utf-8 -*-
"""rc_asm_basisB2_agg.py — 方案B v2 (张量基系数) 全矩阵结果聚合.

读 outputs/asm/basisB2_base_{v}.json (结构化基线) 与
outputs/asm/runs_basisB2/{variant}_{method}/result.json (SR 重组结果),
生成 outputs/asm/t23basisB2_paper/basisB2_summary.json + markdown 表.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "asm"
PAPER = OUT / "t23basisB2_paper"
VARIANTS = ["v", "g", "e", "m"]
METHODS = ["pysr", "lgf_mg", "lgf_go"]
LABEL = {"pysr": "PySR", "lgf_mg": "LGF-mg", "lgf_go": "LGF-go"}


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    summary = {"baselines": {}, "runs": {}, "table": {}}
    for v in VARIANTS:
        p = OUT / f"basisB2_base_{v}.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            summary["baselines"][v] = dict(
                loss_vw=d["loss_vw"], worst=d["worst"], J=d["J"],
                n_params=d["n_params"], per_case=d["per_case"],
                beta_rms=d["beta_rms"], beta_absmax=d.get("beta_absmax"),
                chan_names=d["chan_names"], pool=d["pool"])
    for v in VARIANTS:
        for m in METHODS:
            p = OUT / "runs_basisB2" / f"{v}_{m}" / "result.json"
            if not p.exists():
                continue
            d = json.load(open(p, encoding="utf-8"))
            summary["runs"][f"{v}_{m}"] = dict(
                loss_vw=d["loss_vw"], worst=d["worst"], per_case=d["per_case"],
                J=d["J"], n_targets=len(d["targets"]),
                targets=[dict(target=t["target"], complexity=t["complexity"],
                              loss_coef=t["loss_coef"], dsl=t["dsl"],
                              theta=t.get("theta")) for t in d["targets"]])
    # 表
    rows = {}
    for v in VARIANTS:
        rows[v] = {}
        b = summary["baselines"].get(v)
        if b:
            rows[v]["base"] = round(b["loss_vw"], 4)
        for m in METHODS:
            k = f"{v}_{m}"
            if k in summary["runs"]:
                rows[v][LABEL[m]] = round(summary["runs"][k]["loss_vw"], 4)
    summary["table"] = rows
    json.dump(summary, open(PAPER / "basisB2_summary.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    # markdown
    lines = ["# 方案B v2 (张量基系数) 聚合结果", "",
             "模型: `b23 = Σ_m a_m(Θ)·T̃⁽ᵐ⁾₂₃`, T̃ = 逐工况 RMS 归一张量通道",
             "评估: loss_vw = mean_c[mean_y(b23_pred-b23)²/var_c] (越小越好)", "",
             "| 变体 | 结构化联合解 | PySR | LGF-mg | LGF-go | LGF 最优 |",
             "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r = rows.get(v, {})
        lgfs = [r[k] for k in ("LGF-mg", "LGF-go") if k in r]
        best = min(lgfs) if lgfs else None
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            v, r.get("base", "-"), r.get("PySR", "-"),
            r.get("LGF-mg", "-"), r.get("LGF-go", "-"),
            best if best is not None else "-"))
    (PAPER / "basisB2_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[saved] {PAPER/'basisB2_summary.json'}")


if __name__ == "__main__":
    main()
