# -*- coding: utf-8 -*-
"""vw-only + T23 通道 PySR 快速实验 (2026-10-12 夜)
问题: b23 拟合把张量基 23 分量 (T1/T5/T6/T10) 加进特征池是否有增益?
设计 (每变体):
  ctrl: 纯标量 [eta2,zeta2,s3,so2,s2o2,ret,roloc,yw] (8 维)
  t23 : ctrl + [T1s,T5s,T6s,T10s] (RMS 归一, 12 维)
口径: dataset_trueasm_{v,g,e}, vw 分量, yw∈[0,1], NMSE 分母=逐工况 var_c,
      权重 1/(Nv*var_c) 与 rc_asm_vw_only 同向。
"""
import sys, time, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rc_asm_vw_only import eval_vw_predictions, summarize_vw  # noqa: E402

OUT = Path(r"d:\AI\LLM-Feynman-G\outputs\asm")
ROS = [7.5, 15, 30, 75, 150]
IDX23 = 5
CHANS = ["T1", "T5", "T6", "T10"]
SCAL = ["eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc"]


def load_vw_t23(variant):
    ds = np.load(OUT / f"dataset_trueasm_{variant}.npz")
    cases = []
    for i, ro in enumerate(ROS):
        p = f"c{i}_"
        valid = ds[p + "valid"].astype(bool)
        yh = ds[p + "yh"][valid]
        scal = ds[p + "scalars"][valid]
        yw = 1.0 - np.abs(yh)
        X = np.column_stack([scal, yw])                      # 8 标量
        T23 = np.column_stack([ds[p + m][valid, IDX23] for m in CHANS])
        b23 = ds[p + "b6"][valid, IDX23].astype(float)
        cases.append(dict(i=i, ro=ro, Xfit=X, T23=T23, yfit=b23, yh=yh,
                          Nv=int(valid.sum()), var_c=float(np.var(b23)),
                          max_abs_ref=float(np.max(np.abs(b23)))))
    return cases


def run(tag, Xs, names, cases, maxsize=30, key="Xfit"):
    from pysr import PySRRegressor
    X = np.vstack(Xs); y = np.concatenate([c["yfit"] for c in cases])
    w = np.concatenate([np.full(c["Nv"], 1.0 / (c["Nv"] * c["var_c"])) for c in cases])
    unary = ["exp", "log", "sqrt"]
    model = PySRRegressor(
        niterations=60, populations=8, population_size=40, maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=unary, nested_constraints={u: {v: 0 for v in unary} for u in unary},
        constraints={u: 15 for u in unary},
        model_selection="best", random_state=0,
        temp_equation_file=True, verbosity=0, progress=False)
    t0 = time.time()
    model.fit(X, y, weights=w, variable_names=names)
    print(f"[{tag}] fit done {time.time()-t0:.0f}s, pareto size={len(model.equations_)}", flush=True)
    best, rows = None, []
    for idx in range(len(model.equations_)):
        row = model.equations_.iloc[idx]
        try:
            preds = [np.asarray(model.predict(c[key], index=idx), dtype=float)
                     for c in cases]
            m = eval_vw_predictions(preds, cases)
        except Exception as e:
            print(f"    eq{idx} eval fail: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append((int(row["complexity"]), m["loss_vw"], str(row["sympy_format"])))
        if m.get("per_case") and (best is None or m["loss_vw"] < best[1]):
            best = (idx, m["loss_vw"], m, str(row["sympy_format"]), int(row["complexity"]))
    print(f"[{tag}] {time.time()-t0:.0f}s  best loss_vw={best[1]:.4f} cplx={best[4]}", flush=True)
    print(f"    {summarize_vw(best[2])}", flush=True)
    print(f"    {best[3]}", flush=True)
    return best


def main():
    results = {}
    for variant in ["g", "e", "v"]:
        cases = load_vw_t23(variant)
        # 通道 RMS 归一 (跨工况 pooled)
        rms = np.sqrt(np.mean(np.vstack([c["T23"] for c in cases]) ** 2, axis=0))
        for c in cases:
            c["Xctrl"] = c["Xfit"].copy()
            c["Xt23"] = np.column_stack([c["Xfit"], c["T23"] / rms])
        # ctrl
        best = run(f"ctrl_{variant}", [c["Xctrl"] for c in cases],
                   SCAL + ["yw"], cases, key="Xctrl")
        results[f"ctrl_{variant}"] = dict(loss=best[1], cplx=best[4],
                                          eq=best[3], per_case=best[2]["per_case"])
        # t23
        best = run(f"t23_{variant}", [c["Xt23"] for c in cases],
                   SCAL + ["yw"] + [f"{m}s" for m in CHANS], cases, key="Xt23")
        results[f"t23_{variant}"] = dict(loss=best[1], cplx=best[4],
                                         eq=best[3], per_case=best[2]["per_case"])
    (OUT / "runs_vw_t23_pysr").mkdir(exist_ok=True)
    with open(OUT / "runs_vw_t23_pysr" / "t23_pysr_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    print("\nsaved -> outputs/asm/runs_vw_t23_pysr/t23_pysr_result.json")


if __name__ == "__main__":
    main()
