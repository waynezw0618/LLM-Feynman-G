# -*- coding: utf-8 -*-
"""Aggregate T23-full (v/g/e/m x 12-feature pools) LGF vs PySR results.

Collects:
  LGF  : outputs/asm/runs_vw_t23full_lgf_mg/{v,g,e,m}/result.json  (arm mg)
         outputs/asm/runs_vw_t23full_lgf_go/{v,g,e,m}/result.json  (arm go)
  PySR : outputs/asm/runs_vw_t23full_pysr/{v,g,e,m}/result.json        (seed 0)
         outputs/asm/runs_vw_t23full_pysr/{v,g,e,m}_s{1,2}/result.json (seeds 1,2)

Outputs: outputs/asm/t23full_paper/t23full_summary.json  (+ console markdown table)
Missing runs are tolerated and listed.
"""
import json
import os
import statistics as st
from pathlib import Path

ROOT = Path(r"D:\AI\LLM-Feynman-G")
ASM = ROOT / "outputs" / "asm"
OUTDIR = ASM / "t23full_paper"
VARIANTS = ["v", "g", "e", "m"]
RO_KEYS = ["Ro7.5", "Ro15", "Ro30", "Ro75", "Ro150"]


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed to parse {p}: {e}")
        return None


def grab(d, kind):
    """kind = 'lgf' (entries[0]) or 'pysr' (best)."""
    if kind == "lgf":
        ent = d["entries"][0]
        eq = ent.get("dsl", "")
        cplx = ent.get("complexity", {})
        n_par = cplx.get("params", 0)
        n_nodes = cplx.get("nodes", 0)
    else:
        ent = d["best"]
        eq = ent.get("equation", "")
        cplx = ent.get("complexity", 0)
        n_par = None
        n_nodes = cplx
    m = ent["metrics"]
    preds = d.get("preds_file") or ent.get("preds_file")
    return {
        "formula": eq,
        "loss_vw": m["loss_vw"],
        "worst": m["worst_case_nmse"],
        "per_case": {k: m["per_case"][k]["nmse"] for k in RO_KEYS},
        "complexity_nodes": n_nodes,
        "n_params": n_par,
        "elapsed_s": d.get("elapsed_s"),
        "preds_file": (Path(preds).as_posix() if preds else None),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []

    # ---- LGF ----
    for arm in ["lgf_mg", "lgf_go"]:
        for v in VARIANTS:
            p = ASM / f"runs_vw_t23full_{arm}" / v / "result.json"
            d = load_json(p)
            if d is None:
                missing.append(str(p))
                continue
            r = grab(d, "lgf")
            r.update({"method": arm, "variant": v, "run_dir": str(p.parent)})
            rows.append(r)

    # ---- PySR ----
    for v in VARIANTS:
        for seed, sfx in [(0, ""), (1, "_s1"), (2, "_s2")]:
            p = ASM / "runs_vw_t23full_pysr" / f"{v}{sfx}" / "result.json"
            d = load_json(p)
            if d is None:
                missing.append(str(p))
                continue
            r = grab(d, "pysr")
            r.update({"method": "pysr", "variant": v, "seed": seed,
                      "run_dir": str(p.parent)})
            rows.append(r)

    summary = {"rows": rows, "missing": missing}

    # ---- group stats ----
    def stats(vals):
        return {
            "n": len(vals),
            "mean": st.mean(vals) if vals else None,
            "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    grouped = {}
    for v in VARIANTS:
        grouped[v] = {}
        for meth, sel in [
            ("pysr", [r for r in rows if r["method"] == "pysr" and r["variant"] == v]),
            ("lgf_mg", [r for r in rows if r["method"] == "lgf_mg" and r["variant"] == v]),
            ("lgf_go", [r for r in rows if r["method"] == "lgf_go" and r["variant"] == v]),
            ("lgf_all", [r for r in rows if r["method"].startswith("lgf") and r["variant"] == v]),
            ("all_lgf_vs_pysr", [r for r in rows if r["variant"] == v]),
        ]:
            grouped[v][meth] = {
                "loss_vw": stats([r["loss_vw"] for r in sel]),
                "worst": stats([r["worst"] for r in sel]),
            }
    summary["grouped"] = grouped

    # ---- console markdown ----
    print("\n| variant | PySR seeds (loss_vw) | mean±std | LGF mg | LGF go | LGF mean±std |")
    print("|---|---|---|---|---|---|")
    for v in VARIANTS:
        ps = [f"{r['loss_vw']:.3f}" for r in rows
              if r["method"] == "pysr" and r["variant"] == v]
        g = grouped[v]
        lg = [r["loss_vw"] for r in rows
              if r["method"].startswith("lgf") and r["variant"] == v]
        lg_s = ", ".join(f"{x:.3f}" for x in lg) if lg else "-"
        pm = g["pysr"]["loss_vw"]
        lm = g["lgf_all"]["loss_vw"]
        print(f"| {v} | {' / '.join(ps) if ps else '-'} "
              f"| {pm['mean']:.3f}±{pm['std']:.3f} "
              f"| {g['lgf_mg']['loss_vw']['mean'] if g['lgf_mg']['loss_vw']['n'] else '-'} "
              f"| {g['lgf_go']['loss_vw']['mean'] if g['lgf_go']['loss_vw']['n'] else '-'} "
              f"| {(lm['mean'], lm['std']) if lm['n'] else '-'} |")

    # formatted lgf line fix (mean may be None)
    out = OUTDIR / "t23full_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}  ({len(rows)} runs, {len(missing)} missing)")
    for mfile in missing:
        print(f"   missing: {mfile}")


if __name__ == "__main__":
    main()
