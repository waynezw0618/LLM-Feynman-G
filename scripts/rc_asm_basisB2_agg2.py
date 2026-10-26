# -*- coding: utf-8 -*-
"""rc_asm_basisB2_agg2.py — 方案B2 v2 对照轮聚合.

汇总 runs_basisB2/ 下全部 result.json, 产出:
  t23basisB2_paper/basisB2_v2_summary.json
  控制台分组对照表 (原始 / 等复杂度 / 知识消融)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "asm"
RUNS = OUT / "runs_basisB2"
PAPER = OUT / "t23basisB2_paper"

GROUPS = {
    "raw":      [""],                        # 原始口径 (PySR ms25 / LGF 旧配置)
    "ms12":     ["_ms12"],                   # 等复杂度 PySR (maxsize=12)
    "v2":       ["_v2"],                     # LGF 增强 (physics 提示)
    "nophys":   ["_nophys"],                 # LGF 知识消融 (generic 提示)
    "big":      ["_big"],                    # LGF 大形状+结构提示
}
VARIANTS = ("v", "g", "e", "m")
METHODS = ("pysr", "lgf_mg", "lgf_go")


def load(tag):
    rows = {}
    for v in VARIANTS:
        for m in METHODS:
            p = RUNS / f"{v}_{m}{tag}" / "result.json"
            if not p.exists():
                continue
            d = json.load(open(p, encoding="utf-8"))
            rf = d.get("refit") or {}
            rows[f"{v}_{m}"] = dict(
                loss=d["loss_vw"], worst=d["worst"], J=d["J"],
                cplx=[t["complexity"] for t in d.get("targets", [])],
                loss_coef=[t["loss_coef"] for t in d.get("targets", [])],
                per_case=d.get("per_case", {}),
                seed=d.get("seed", 0),
                refit_loss=rf.get("loss_vw"), refit_worst=rf.get("worst"),
                refit_npar=rf.get("n_params"),
                refit_per_case=rf.get("per_case", {}))
    return rows


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    summary = {}
    for gname, tags in GROUPS.items():
        for tag in tags:
            rows = load(tag)
            if rows:
                summary[f"{gname}{tag}"] = rows

    json.dump(summary, open(PAPER / "basisB2_v2_summary.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    for key, rows in summary.items():
        print(f"===== {key} =====")
        for k in sorted(rows):
            r = rows[k]
            rf = (f"  refit={r['refit_loss']:.4g}(n={r['refit_npar']})"
                  if r.get("refit_loss") is not None else "")
            print(f"  {k:12s} loss={r['loss']:>10.4g}  worst={r['worst']:>9.3g}"
                  f"{rf}")
    print(f"\n-> {PAPER/'basisB2_v2_summary.json'}")


if __name__ == "__main__":
    main()
