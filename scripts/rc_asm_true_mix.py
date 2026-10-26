# -*- coding: utf-8 -*-
"""rc_asm_true_mix.py — 真 ASM 混合 (MIX) 实验.

目的 (用户 2026-10-09 指令):
  把 v / g / e 三个变体的 T1..T10 与 5 个标量不变量混合成 43 维特征空间,
  让 PySR 和 LGF-go 在同一份"全要素"输入上竞争, 验证/证伪下述假设:
    假设: v 的 W̄ 是 RANS 教材的"母语量"; g/e 把 Ω 塞进 W̄ 是混血冗余.
    预测: 混合训练中 v 的 T2_v/T4_v/... 应被选中, g/e 的 T2_g/T4_e/...
          应被赋 0 或极小系数.

特征布局 (43 列):
  T1 (1) = S                        <-- 三变体共享
  T2 (3) = [S, W_v] / [S, W_g] / [S, W_e]
  T3 (1) = S^2                      <-- 共享
  T4..T10 (3 each = 21) = 各自 W 构造
  eta2/zeta2/s3/so2/s2o2 (3 each = 15) = 各自 W 构造
  ret, roloc (2)                     <-- 共享
  TOTAL = 2 + 3 + 1 + 21 + 15 + 2 = 43

yfit (b_ij) = DNS 真值, 与变体无关 (用 v 的就行).

输出:
  outputs/asm/dataset_trueasm_mix.npz          (混合特征数据集)
  outputs/asm/runs_trueasm_mix_pysr/result.json  (PySR Pareto + best)
  outputs/asm/runs_trueasm_mix_lgf_go/result.json (LGF-go 40 代 best)
  outputs/asm/_trueasm_mix_pysr.log
  outputs/asm/_trueasm_mix_lgf_go.log

用法:
  python -X utf8 scripts/rc_asm_true_mix.py
  python -X utf8 scripts/rc_asm_true_mix.py --skip_pysr
  python -X utf8 scripts/rc_asm_true_mix.py --skip_lgf
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rc_asm_true_eval import (RO_CASES, eval_predictions, eval_case,
                              rho_a_profile, _full_tensor, COMP6,
                              load_variant, lgf_predict, summarize)
from rc_asm_true_prep import FEAT_NAMES
from rc_asm_run import BACKENDS
from lgf.config import LGFConfig
from lgf import llm_ops
from lgf import dsl as lgfdsl
from lgf.pipeline_lgf import run_lgf

OUT_DIR = REPO / "outputs" / "asm"

# T_ij 在 v/g/e 下哪些是"共享的" (S 或 S^2 不含 W)
SHARED_T = {0: "T1", 2: "T3"}    # col_idx 0=T1, 2=T3
# T_ij 在 v/g/e 下是"分裂的" (W 不同 → 这些基的数值不同)
SPLIT_T = {1, 3, 4, 5, 6, 7, 8, 9}    # T2, T4..T10
SPLIT_INV = {"eta2", "zeta2", "s3", "so2", "s2o2"}
SHARED_SCAL = {"ret", "roloc"}


def build_mix_feat_names():
    """43 列特征名, 按 [T1, T2_v/g/e, T3, T4..T10_v/g/e, eta2.._v/g/e, ret, roloc] 排."""
    names = []
    for ti in range(1, 11):  # T1..T10
        if ti - 1 in SHARED_T:
            names.append(f"T{ti}")
        else:
            for v in ("v", "g", "e"):
                names.append(f"T{ti}_{v}")
    for inv in ("eta2", "zeta2", "s3", "so2", "s2o2"):
        for v in ("v", "g", "e"):
            names.append(f"{inv}_{v}")
    names.append("ret")
    names.append("roloc")
    assert len(names) == 43, len(names)
    return names


def build_mix_dataset(out_path: Path):
    """读 v/g/e 三个 npz, 拼成 43 列混合特征 npz."""
    feat_names = build_mix_feat_names()
    # 列索引映射: feat_name -> 整列 (从对应 v/g/e Xfit 抽取)
    col_layout = []
    for ti in range(1, 11):
        if ti - 1 in SHARED_T:
            col_layout.append(("any", ti - 1))           # 共享列, 任一变体都行
        else:
            for v in ("v", "g", "e"):
                col_layout.append((v, ti - 1))           # 分裂列
    for inv in ("eta2", "zeta2", "s3", "so2", "s2o2"):
        for v in ("v", "g", "e"):
            col_layout.append((v, ["eta2", "zeta2", "s3", "so2", "s2o2"].index(inv) + 10))
    col_layout.append(("any", 15))  # ret
    col_layout.append(("any", 16))  # roloc
    assert len(col_layout) == 43

    ds_v = np.load(OUT_DIR / "dataset_trueasm_v.npz")
    ds_g = np.load(OUT_DIR / "dataset_trueasm_g.npz")
    ds_e = np.load(OUT_DIR / "dataset_trueasm_e.npz")
    dss = {"v": ds_v, "g": ds_g, "e": ds_e}

    save = dict(feat_names=np.array(feat_names), variant=np.array("mix"),
                variant_label=np.array(
                    "MIX: v+g+e T_ij (T1/T3 shared, T2,T4..T10 split×3) + "
                    "5 invariants × 3 + ret/roloc = 43 cols"),
                ref=np.array("Same as v/g/e dataset, just feature-expanded; "
                             "y = b_ij (DNS) is variant-independent."))
    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        save[p + "ro"] = ds_v[p + "ro"]
        save[p + "yh"] = ds_v[p + "yh"]
        save[p + "k"] = ds_v[p + "k"]
        save[p + "eps"] = ds_v[p + "eps"]
        save[p + "valid"] = ds_v[p + "valid"]
        save[p + "b6"] = ds_v[p + "b6"]
        valid = ds_v[p + "valid"]
        # ---- 拼 Xfit: 每行 43 列, 行 = 分量 * Nv + j (分量优先) ----
        Nv = int(valid.sum())
        Ntot = 6 * Nv
        Xmix = np.zeros((Ntot, 43), dtype=float)
        for col, (var_key, src_col) in enumerate(col_layout):
            Xfull = dss[var_key if var_key != "any" else "v"][p + "Xfit"]
            Xmix[:, col] = Xfull[:, src_col]
        save[p + "Xfit"] = Xmix
        save[p + "yfit"] = ds_v[p + "yfit"]
        # scalars 也得拼, 用于 eval 反推
        save[p + "scalars"] = ds_v[p + "scalars"]
        print(f"[mix-prep] Ro={ro:>5}: Nv={Nv}, Xfit={Xmix.shape}, "
              f"|T1|max={np.abs(Xmix[:,0]).max():.3g}, "
              f"|T2_v|_max={np.abs(Xmix[:,1]).max():.3g}, "
              f"|T2_g|_max={np.abs(Xmix[:,2]).max():.3g}, "
              f"|T2_e|_max={np.abs(Xmix[:,3]).max():.3g}")
    np.savez_compressed(out_path, **save)
    print(f"[mix-prep] -> {out_path}  (43 features)")
    return feat_names


def load_mix():
    """读 mix npz, 5 工况拟合行数据 (与 load_variant 同结构, 但 Xfit 43 列)."""
    ds = np.load(OUT_DIR / "dataset_trueasm_mix.npz")
    cases = []
    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        valid = ds[p + "valid"]
        Xfit = ds[p + "Xfit"]
        yfit = ds[p + "yfit"]
        Nv = int(valid.sum())
        assert yfit.shape[0] == 6 * Nv
        cases.append(dict(
            i=i, ro=float(ro), Xfit=Xfit, yfit=yfit, Nv=Nv,
            b6v=ds[p + "b6"][valid].astype(float),
            yh=ds[p + "yh"][valid].astype(float),
            var_c=float(np.var(yfit)),
            max_abs_ref=float(np.max(np.abs(yfit)))))
    return cases


def lgf_predict_mix(dsl: str, theta, data: list, var_names: list):
    """重编译 DSL + theta → 每工况预测行. var_names 是 43 维混合特征名."""
    from lgf.compiler import CompiledGraph
    g = lgfdsl.parse_dsl(dsl, var_names)
    if g.case_names():
        raise ValueError(f"DSL 含逐工况常数 K: {g.case_names()} (真 ASM 禁止)")
    cg = CompiledGraph(g, backend="numpy")
    theta = np.asarray(theta, dtype=float)
    preds = []
    for case in data:
        preds.append(np.asarray(cg.eval_case(theta, case["Xfit"]), dtype=float))
    return preds, var_names


# ============================================================================
# 1) PySR 跑批
# ============================================================================
def run_pysr(var_names, data, niterations=60, maxsize=30, populations=8,
             pop_size=40, seed=0, unary=None, run_dir=None):
    from pysr import PySRRegressor
    X = np.vstack([c["Xfit"] for c in data])
    y = np.concatenate([c["yfit"] for c in data])
    w = np.concatenate([
        np.full(6 * c["Nv"], 1.0 / (6.0 * c["Nv"] * c["var_c"])) for c in data])
    print(f"[mix-pysr] X={X.shape} y={y.shape}")
    unary = unary or ["exp", "log", "sqrt"]
    nested = {u: {v: 0 for v in unary} for u in unary}
    model = PySRRegressor(
        niterations=niterations, populations=populations,
        population_size=pop_size, maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=unary, nested_constraints=nested,
        constraints={u: 15 for u in unary},
        model_selection="best", random_state=seed,
        temp_equation_file=True, verbosity=1, progress=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.fit(X, y, weights=w, variable_names=var_names)
    elapsed = time.time() - t0
    print(f"[mix-pysr] fit done in {elapsed:.0f}s, Pareto size={len(model.equations_)}")
    pareto = []
    eqs = model.equations_
    for idx in range(len(eqs)):
        row = eqs.iloc[idx]
        try:
            preds = [np.asarray(model.predict(c["Xfit"], index=idx), dtype=float)
                     for c in data]
            m = eval_predictions(preds, data)
        except Exception as e:
            m = dict(loss_asm=float("inf"), rho_a=-1.0, per_case={},
                     error=f"{type(e).__name__}: {e}")
        rec = dict(index=idx, complexity=int(row["complexity"]),
                   internal_loss=float(row["loss"]),
                   sympy=str(row.get("sympy_format", row["equation"])),
                   equation=str(row["equation"]), metrics=m)
        pareto.append(rec)
        print(f"[mix-pysr] eq{idx} cplx={int(row['complexity']):<3} "
              f"internal={float(row['loss']):.5f} :: "
              f"{summarize(m) if m.get('per_case') else m.get('error')}")
    finite = [r for r in pareto if np.isfinite(r["metrics"].get("loss_asm", np.inf))]
    best = min(finite, key=lambda r: r["metrics"]["loss_asm"]) if finite else None
    result = dict(variant="mix", method="pysr", n_features=len(var_names),
                  feat_names=list(var_names),
                  config=dict(niterations=niterations, maxsize=maxsize,
                              populations=populations, population_size=pop_size,
                              unary=unary, seed=seed),
                  n_rows=int(X.shape[0]), pareto=pareto,
                  elapsed_s=round(elapsed, 1))
    if best is not None:
        result["best"] = dict(index=best["index"], complexity=best["complexity"],
                              internal_loss=best["internal_loss"],
                              sympy=best["sympy"], equation=best["equation"],
                              metrics=best["metrics"])
        preds = [np.asarray(model.predict(c["Xfit"], index=best["index"]),
                            dtype=float) for c in data]
        preds_file = run_dir / "preds_best.npz"
        np.savez_compressed(preds_file, **{f"c{i}": p for i, p in enumerate(preds)})
        result["preds_file"] = str(preds_file)
        print(f"[mix-pysr] BEST eq{best['index']} cplx={best['complexity']}\n"
              f"  {best['sympy']}\n  {summarize(best['metrics'])}")
    out = run_dir / "result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[mix-pysr] -> {out}")
    return result


# ============================================================================
# 2) LGF-go 跑批
# ============================================================================
def run_lgf_go(var_names, data, gens=40, pop=60, budget=900, sub_sample=300,
               max_edges=16, max_params=6, seed=0, run_dir=None):
    be = BACKENDS["minimax"]
    api_key = os.environ.get("LGF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[mix-lgf] LLM API key missing; abort.")
    client = llm_ops.make_client(be["backend"], url=be["url"], model=be["model"],
                                 api_key=api_key, timeout=240.0)
    cases = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}") for c in data]
    print(f"[mix-lgf-go] cases={[c['name'] for c in cases]} feats={len(var_names)}")
    import zlib
    s = (zlib.crc32(b"trueasm|mix|go") + seed) % (2**31)
    cfg = LGFConfig(
        population=pop, generations=gens, sub_sample=sub_sample,
        time_budget_s=budget, max_edges=max_edges, max_params=max_params,
        final_starts=8, final_maxiter=150,
        llm_backend=be["backend"], llm_url=be["url"], llm_model=be["model"],
        llm_api_key=api_key, llm_timeout=240.0, seed=s,
        llm_ginit_only=True, wall_seeds=False,
        physics_context=PHYSICS_CONTEXT_MIX,
        global_params_only=True,
        ginit_retry=2, ginit_min_kept=3, ginit_fewshot=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = "trueasm_mix_go"
    t0 = time.time()
    res = run_lgf(cases, var_names, cfg, run_dir, client, name=name)
    elapsed = time.time() - t0
    entries = [("best", res["best"])] + [
        (f"top{i}", e) for i, e in enumerate(res.get("top_final", []))]
    evaluated = []
    for tag, e in entries:
        rec = dict(tag=tag, dsl=e["dsl"], loss_paramfit=e["loss"],
                   complexity=e["complexity"], theta=e["theta"],
                   n_g=len(lgfdsl.parse_dsl(e["dsl"], var_names).global_names()),
                   n_c=len(lgfdsl.parse_dsl(e["dsl"], var_names).case_names()))
        try:
            preds, _ = lgf_predict_mix(e["dsl"], e["theta"], data, var_names)
            rec["metrics"] = eval_predictions(preds, data)
            if tag == "best":
                preds_file = run_dir / "preds_best.npz"
                np.savez_compressed(preds_file, **{f"c{i}": p for i, p in enumerate(preds)})
                rec["preds_file"] = str(preds_file)
        except Exception as ex:
            rec["metrics"] = dict(error=f"{type(ex).__name__}: {ex}")
        evaluated.append(rec)
        m = rec["metrics"]
        print(f"[mix-lgf-go] {tag}: cplx={rec['complexity']} loss={e['loss']:.5f} :: "
              f"{summarize(m) if m.get('per_case') else m.get('error')}")
    summary = dict(variant="mix", arm="go", method="lgf",
                   backend="minimax", model=be["model"], seed=s,
                   config=dict(budget=budget, pop=pop, gens=gens,
                               sub_sample=sub_sample, max_edges=max_edges,
                               max_params=max_params,
                               llm_ginit_only=True,
                               global_params_only=True,
                               n_features=len(var_names)),
                   feat_names=list(var_names), elapsed_s=round(elapsed, 1),
                   llm_stats=res.get("llm_stats", {}),
                   entries=evaluated)
    out = run_dir / "result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[mix-lgf-go] -> {out}")
    return summary


PHYSICS_CONTEXT_MIX = """ASM (代数应力模型) b_ij = <u_i'u_j'>/(2k) - delta_ij/3, 旋转槽道流 DNS,
Re_τ=180, Ro_τ=7.5/15/30/75/150 (Ω 沿流向 x1).
本任务是 MIX 训练: 同时提供 v / g / e 三种张量基的 T1..T10, 算法应在 Pareto 竞争中
"投票"选出最有用的子集.
特征命名约定:
- T1, T3: 三变体共享 (基于 S 的对称函数, 不含 W), 单一列
- T2 / T4..T10: 三变体分裂为 T#_v / T#_g / T#_e, 三列一组
  · T#_v: W = W̄ (标准 RANS 闭包量)
  · T#_g: W = W̄ + 1·Ω (Gatski-Speziale 1993, 中间量)
  · T#_e: W = W̄ + 2·Ω (扩展内禀, ω_a)
- eta2/zeta2/s3/so2/s2o2 同样分裂三列
- ret, roloc 共享 (二标量)
总特征维 = 43.
物理提示: 之前单变体实验已经显示 v 的 W̄ 是 RANS 教材的"母语量", 算法更易在 v 的
T 集上找到 f(T1..T10) 的简洁形式; g/e 的 W̄+rfΩ 含义模糊, 通常被 SR 算法赋小系数或拒绝.
推荐结构: b_ij = Σ_{m, v ∈ {v,g,e}} g_m(I_v) · T_m_v  (v 通道应被赋大系数).
硬规则: 仅允许全局常数 C#; 禁逐工况 K# (违反 ASM 普适性)."""


# ============================================================================
# 3) 简易汇总 (10 个 run + 1 个 mix 的 head-to-head)
# ============================================================================
def compare_to_baseline():
    """把 mix vs 9 个原 run 的 loss_asm / rho_a 摆一起, 给论文用."""
    def _scalar(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    runs = {}
    for v in ("v", "g", "e"):
        for arm in ("mg", "go"):
            p = OUT_DIR / "runs_trueasm_lgf" / f"{v}_{arm}" / "result.json"
            if p.exists():
                j = json.load(open(p, encoding="utf-8"))
                best = j["entries"][0]
                m = best.get("metrics", {}) or {}
                runs[f"lgf_{v}_{arm}"] = dict(
                    loss=_scalar(m.get("loss_asm", float("inf"))),
                    rho=_scalar(m.get("rho_a", float("nan"))),
                    dsl=best.get("dsl", ""),
                    complexity=_scalar(best.get("complexity", 0)))
        p = OUT_DIR / "runs_trueasm_pysr" / v / "result.json"
        if p.exists():
            j = json.load(open(p, encoding="utf-8"))
            b = j.get("best", {}) or {}
            m = b.get("metrics", {}) or {}
            runs[f"pysr_{v}"] = dict(
                loss=_scalar(m.get("loss_asm", float("inf"))),
                rho=_scalar(m.get("rho_a", float("nan"))),
                dsl=b.get("equation", ""),
                complexity=_scalar(b.get("complexity", 0)))
    # mix
    for k, sub in [("mix_pysr", "runs_trueasm_mix_pysr"),
                   ("mix_lgf_go", "runs_trueasm_mix_lgf_go")]:
        p = OUT_DIR / sub / "result.json"
        if p.exists():
            j = json.load(open(p, encoding="utf-8"))
            if "best" in j and "pareto" in j:    # pysr
                m = j["best"]["metrics"] or {}
                runs[k] = dict(loss=_scalar(m.get("loss_asm", float("inf"))),
                               rho=_scalar(m.get("rho_a", float("nan"))),
                               dsl=j["best"].get("sympy", ""),
                               complexity=_scalar(j["best"].get("complexity", 0)))
            elif "entries" in j:                # lgf
                b = j["entries"][0]
                m = b.get("metrics", {}) or {}
                runs[k] = dict(loss=_scalar(m.get("loss_asm", float("inf"))),
                               rho=_scalar(m.get("rho_a", float("nan"))),
                               dsl=b.get("dsl", ""),
                               complexity=_scalar(b.get("complexity", 0)))
    out = OUT_DIR / "compare_with_mix.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n=== 9 原 run + mix 对比 ===")
    print(f"{'run':<22} {'loss_asm':>10} {'rho_a':>8} {'cplx':>5}")
    for k in ["lgf_v_go", "lgf_g_go", "lgf_e_go", "lgf_v_mg", "lgf_g_mg", "lgf_e_mg",
              "pysr_v", "pysr_g", "pysr_e", "mix_lgf_go", "mix_pysr"]:
        if k in runs:
            r = runs[k]
            print(f"{k:<22} {r['loss']:>10.4f} {r['rho']:>+8.4f} {r['complexity']:>5.0f}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_prep", action="store_true")
    ap.add_argument("--skip_pysr", action="store_true")
    ap.add_argument("--skip_lgf", action="store_true")
    ap.add_argument("--pysr_iters", type=int, default=60)
    ap.add_argument("--lgf_gens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz_path = OUT_DIR / "dataset_trueasm_mix.npz"
    if args.skip_prep and npz_path.exists():
        print(f"[mix] skip prep, re-using {npz_path}")
    else:
        feat_names = build_mix_dataset(npz_path)
    feat_names = build_mix_feat_names()
    print(f"[mix] feat_names ({len(feat_names)}): {feat_names}")

    data = load_mix()
    print(f"[mix] data loaded: {[c['Nv'] for c in data]} valid pts/case, "
          f"Xfit shape/case = {data[0]['Xfit'].shape}")

    pysr_dir = OUT_DIR / "runs_trueasm_mix_pysr"
    lgf_dir = OUT_DIR / "runs_trueasm_mix_lgf_go"

    if not args.skip_pysr:
        print("\n========== PySR (40 迭代) ==========")
        run_pysr(feat_names, data, niterations=args.pysr_iters, maxsize=30,
                 populations=8, pop_size=40, seed=args.seed, run_dir=pysr_dir)
    if not args.skip_lgf:
        print("\n========== LGF-go (40 代) ==========")
        run_lgf_go(feat_names, data, gens=args.lgf_gens, pop=60, budget=900,
                   sub_sample=300, max_edges=16, max_params=6, seed=args.seed,
                   run_dir=lgf_dir)

    compare_to_baseline()


if __name__ == "__main__":
    main()
