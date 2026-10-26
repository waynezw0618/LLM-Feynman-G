# -*- coding: utf-8 -*-
"""rc_asm_vw_only.py — 只拟合雷诺应力 <vw> 分量 (b23).

背景: 用户实际应用里涡量输运方程 (2) 只用到 <vw>; 全局 6 分量 loss 被 b11/b22 等
大尺度分量淹没, 导致 b23 单独 NMSE 高达 0.5-0.8. 本脚本把问题退化为单输出
符号回归, 目标仅是 b23(y).

覆盖矩阵 (variant × method):
  variant = {v, g, e, m}
  method  = {pysr, lgf_go, lgf_mg}
    lgf_go: ginit_only=True  (LLM 只出初始种群, 进化离线)
    lgf_mg: ginit_only=False (完整 M_GUIDE)

数据:
  v/g/e: dataset_trueasm_{v,g,e}.npz  (17 特征)
  m    : dataset_trueasm_mix.npz       (43 特征)

输出:
  outputs/asm/runs_vwonly_{method}/{variant}/result.json

用法示例:
  # PySR
  python -X utf8 scripts/rc_asm_vw_only.py --variant v --method pysr
  # LGF go
  python -X utf8 scripts/rc_asm_vw_only.py --variant m --method lgf_go --backend heuristic
  # LGF mg (minimax)
  python -X utf8 scripts/rc_asm_vw_only.py --variant e --method lgf_mg --backend minimax
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rc_asm_true_eval import (RO_CASES, COMP6, load_variant, summarize)
from rc_asm_true_mix import load_mix, build_mix_feat_names, lgf_predict_mix
from rc_asm_run import BACKENDS
from rc_asm_true_prep import FEAT_NAMES
from lgf.config import LGFConfig
from lgf import llm_ops
from lgf import dsl as lgfdsl
from lgf.pipeline_lgf import run_lgf

OUT_DIR = REPO / "outputs" / "asm"
VW_IDX = COMP6.index("vw")  # 5


def extract_vw_only(data: list) -> list:
    """把 6 分量行展开数据切出 vw 分量.

    原数据行顺序: comp*Nv + j (comp=0..5, j=0..Nv-1).
    vw 对应 comp=5, 行 [5*Nv : 6*Nv].
    """
    out = []
    for c in data:
        Nv = c["Nv"]
        s = VW_IDX * Nv
        e = s + Nv
        out.append(dict(
            i=c["i"], ro=c["ro"],
            Xfit=np.asarray(c["Xfit"][s:e], dtype=float),
            yfit=np.asarray(c["yfit"][s:e], dtype=float),
            Nv=Nv,
            b6v=np.asarray(c["b6v"][:, VW_IDX], dtype=float),
            yh=np.asarray(c["yh"], dtype=float),
            var_c=float(np.var(c["yfit"][s:e])),
            max_abs_ref=float(np.max(np.abs(c["yfit"][s:e]))),
        ))
    return out


def load_vw_data(variant: str, add_yw: bool = False, yw_scale: float = 1.0):
    """读对应 npz 并切出 vw 分量.

    add_yw=True 时在特征池末尾追加 yw = (1-|yh|)*yw_scale.

    V4 (2026-10-12) 标准化: 旧版 yw=(1-|yh|)·180 ∈[0,180] 壁面单位, 实测
    把 paramfit θ 压到 ~1e-7 量级 (对数尺度多 start 也难救), 属特征尺度病.
    改为半槽宽无量纲化 yw∈[0,1]: 0=壁面, 1=槽道中心. 对表达力无损失
    (yw/180 与 yw 只差常数因子, log/比值结构中可被 C# 吸收), PySR/LGF
    同池同尺度, 对比公平性不变.

    尺度对照 (V4c): yw_scale=1.0 → ∈[0,1] (本轮主口径); yw_scale=180.0 →
    ∈[0,180] 壁面单位 (上一轮 yw 轮口径, PySR SOTA 0.0329 所用尺度).
    两尺度同池同用, 用于验证结论不依赖尺度选择.
    """
    if variant == "m":
        data = load_mix()
        var_names = build_mix_feat_names()
    else:
        data = load_variant(variant)
        var_names = [str(x) for x in FEAT_NAMES]
    vw = extract_vw_only(data)
    if add_yw:
        for c in vw:
            yw = (1.0 - np.abs(c["yh"])) * float(yw_scale)
            c["Xfit"] = np.column_stack([c["Xfit"], yw])
        var_names = list(var_names) + ["yw"]
    return vw, var_names


def lgf_predict_vw(dsl: str, theta, data: list, var_names: list):
    """重编译 DSL + theta → 每工况 vw 预测 (Nv,)."""
    from lgf.compiler import CompiledGraph
    g = lgfdsl.parse_dsl(dsl, var_names)
    if g.case_names():
        raise ValueError(f"DSL 含逐工况常数 K: {g.case_names()}")
    cg = CompiledGraph(g, backend="numpy")
    theta = np.asarray(theta, dtype=float)
    return [np.asarray(cg.eval_case(theta, c["Xfit"]), dtype=float) for c in data]


def eval_vw_predictions(preds: list, data: list) -> dict:
    """单输出 b23 评估: 逐工况 NMSE + 汇总."""
    per_case = []
    for pred, case in zip(preds, data):
        y = case["yfit"]
        p = np.asarray(pred, dtype=float).reshape(-1)
        if p.shape[0] != case["Nv"]:
            raise ValueError(f"pred len {p.shape[0]} != Nv={case['Nv']}")
        if not np.all(np.isfinite(p)):
            per_case.append({
                "ro": case["ro"], "nmse": float("inf"),
                "max_abs_pred": float("inf"), "frac_oor": 1.0})
            continue
        mse = float(np.mean((p - y) ** 2))
        nmse = mse / (case["var_c"] + 1e-300)
        thr = 20.0 * (case["max_abs_ref"] + 1e-12)
        per_case.append({
            "ro": case["ro"], "nmse": nmse,
            "max_abs_pred": float(np.max(np.abs(p))),
            "frac_oor": float(np.mean(np.abs(p) > thr))})
    nmses = [m["nmse"] for m in per_case]
    return dict(
        loss_vw=float(np.mean(nmses)),
        worst_case_nmse=float(np.max(nmses)),
        max_abs_pred=float(np.max([m["max_abs_pred"] for m in per_case])),
        frac_oor=float(np.max([m["frac_oor"] for m in per_case])),
        per_case={f"Ro{m['ro']:g}": m for m in per_case})


def summarize_vw(metrics: dict) -> str:
    pc = " ".join(f"Ro{k[2:]}:{v['nmse']:.4f}"
                  for k, v in metrics["per_case"].items())
    return (f"loss_vw={metrics['loss_vw']:.4f} "
            f"worst={metrics['worst_case_nmse']:.4f} | {pc}")


# ============================================================================
# PySR
# ============================================================================
def run_pysr_vw(variant: str, data: list, var_names: list, *,
                niterations: int = 60, maxsize: int = 30,
                populations: int = 8, population_size: int = 40,
                seed: int = 0, unary: list = None, run_dir: Path = None):
    from pysr import PySRRegressor

    X = np.vstack([c["Xfit"] for c in data])
    y = np.concatenate([c["yfit"] for c in data])
    # 逐工况方差归一化加权, 与 LGF paramfit 的 loss 同向
    w = np.concatenate([
        np.full(c["Nv"], 1.0 / (c["Nv"] * c["var_c"])) for c in data])
    print(f"[vw-pysr:{variant}] X={X.shape} y={y.shape} "
          f"rows/case={[c['Nv'] for c in data]}")

    unary = unary or ["exp", "log", "sqrt"]
    nested = {u: {v: 0 for v in unary} for u in unary}
    model = PySRRegressor(
        niterations=niterations, populations=populations,
        population_size=population_size, maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=unary, nested_constraints=nested,
        constraints={u: 15 for u in unary},
        model_selection="best", random_state=seed,
        temp_equation_file=True, verbosity=1, progress=False)

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.fit(X, y, weights=w, variable_names=var_names)
    elapsed = time.time() - t0
    print(f"[vw-pysr:{variant}] fit done in {elapsed:.0f}s, "
          f"Pareto front size={len(model.equations_)}")

    pareto = []
    eqs = model.equations_
    for idx in range(len(eqs)):
        row = eqs.iloc[idx]
        try:
            preds = [np.asarray(model.predict(c["Xfit"], index=idx), dtype=float)
                     for c in data]
            m = eval_vw_predictions(preds, data)
        except Exception as e:
            m = dict(loss_vw=float("inf"), per_case={},
                     error=f"{type(e).__name__}: {e}")
            preds = None
        rec = dict(index=idx, complexity=int(row["complexity"]),
                   internal_loss=float(row["loss"]),
                   sympy=str(row.get("sympy_format", row["equation"])),
                   equation=str(row["equation"]), metrics=m)
        pareto.append(rec)
        print(f"[vw-pysr:{variant}] eq{idx} cplx={int(row['complexity']):<3} "
              f"internal={float(row['loss']):.5f} :: "
              f"{summarize_vw(m) if m.get('per_case') else m.get('error')}")

    finite = [r for r in pareto if np.isfinite(r["metrics"].get("loss_vw", np.inf))]
    best = min(finite, key=lambda r: r["metrics"]["loss_vw"]) if finite else None

    result = dict(
        variant=variant, method="pysr",
        target="vw", target_label="b23/<vw>",
        config=dict(niterations=niterations, maxsize=maxsize,
                    populations=populations, population_size=population_size,
                    unary=unary, seed=seed,
                    weights="1/(Nv_c*var_c_vw)"),
        n_features=len(var_names), feat_names=list(var_names),
        n_rows=int(X.shape[0]), elapsed_s=round(elapsed, 1),
        pareto=pareto)
    if best is not None:
        result["best"] = dict(
            index=best["index"], complexity=best["complexity"],
            internal_loss=best["internal_loss"],
            sympy=best["sympy"], equation=best["equation"],
            metrics=best["metrics"])
        preds = [np.asarray(model.predict(c["Xfit"], index=best["index"]),
                            dtype=float) for c in data]
        preds_file = run_dir / "preds_best.npz"
        np.savez_compressed(preds_file, **{f"c{i}": p for i, p in enumerate(preds)})
        result["preds_file"] = str(preds_file)
        print(f"[vw-pysr:{variant}] BEST eq{best['index']} "
              f"cplx={best['complexity']}\n  {best['sympy']}\n  "
              f"{summarize_vw(best['metrics'])}")

    out = run_dir / "result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[vw-pysr:{variant}] -> {out}")
    return result


# ============================================================================
# LGF
# ============================================================================
def run_lgf_vw(variant: str, data: list, var_names: list, *,
               method: str, backend: str, budget: float = 600.0,
               pop: int = 60, gens: int = 40, sub_sample: int = 300,
               max_edges: int = 16, max_params: int = 6,
               seed: int = 0, run_dir: Path = None):
    be = BACKENDS[backend]
    api_key = os.environ.get("LGF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if backend != "heuristic" and not api_key:
        raise SystemExit(f"[vw-lgf:{variant}] backend={backend} 需 LGF_LLM_API_KEY")
    # V4 (2026-10-12): glm-5.3 强制思考且 reasoning_effort 默认 max — 必须显式
    # 降 low + max_tokens 16384 (BACKENDS.glm53 已带 extra_body; 实测 low 档
    # G_INIT 30 候选 13s, 解析率 29/30; 不降档时 reasoning 8181 吃光 8192,
    # content 为空 finish=length).
    client = llm_ops.make_client(be["backend"], url=be["url"], model=be["model"],
                                 api_key=api_key, timeout=300.0,
                                 max_tokens=be.get("max_tokens", 8192),
                                 extra_body=be.get("extra_body"))

    cases = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}") for c in data]
    print(f"[vw-lgf:{variant}] cases={[c['name'] for c in cases]} "
          f"feats={len(var_names)} method={method} backend={backend}")

    ginit_only = (method == "lgf_go")
    import zlib
    s = (zlib.crc32(f"vwonly|{variant}|{method}".encode()) + seed) % (2**31)
    cfg = LGFConfig(
        population=pop, generations=gens, sub_sample=sub_sample,
        time_budget_s=budget, max_edges=max_edges, max_params=max_params,
        final_starts=8, final_maxiter=150,
        llm_backend=be["backend"], llm_url=be["url"], llm_model=be["model"],
        llm_api_key=api_key, llm_timeout=240.0, seed=s,
        llm_ginit_only=ginit_only, wall_seeds=False,
        physics_context=physics_context_vw_for(variant, var_names),
        global_params_only=True,
        ginit_retry=2, ginit_min_kept=3, ginit_fewshot=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"vwonly_{variant}_{method}"
    t0 = time.time()
    res = run_lgf(cases, var_names, cfg, str(run_dir), client, name=name)
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
            if variant == "m":
                preds, _ = lgf_predict_mix(e["dsl"], e["theta"], data, var_names)
            else:
                preds = lgf_predict_vw(e["dsl"], e["theta"], data, var_names)
            rec["metrics"] = eval_vw_predictions(preds, data)
            if tag == "best":
                preds_file = run_dir / "preds_best.npz"
                np.savez_compressed(preds_file, **{f"c{i}": p for i, p in enumerate(preds)})
                rec["preds_file"] = str(preds_file)
        except Exception as ex:
            rec["metrics"] = dict(error=f"{type(ex).__name__}: {ex}")
        evaluated.append(rec)
        m = rec["metrics"]
        print(f"[vw-lgf:{variant}] {tag}: cplx={rec['complexity']} "
              f"loss={e['loss']:.5f} :: "
              f"{summarize_vw(m) if m.get('per_case') else m.get('error')}")

    summary = dict(
        variant=variant, method=method, target="vw", target_label="b23/<vw>",
        backend=backend, model=be["model"], seed=s,
        config=dict(budget=budget, pop=pop, gens=gens,
                    sub_sample=sub_sample, max_edges=max_edges,
                    max_params=max_params, llm_ginit_only=ginit_only,
                    global_params_only=True, n_features=len(var_names)),
        feat_names=list(var_names), elapsed_s=round(elapsed, 1),
        llm_stats=res.get("llm_stats", {}),
        entries=evaluated)
    out = run_dir / "result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[vw-lgf:{variant}] -> {out}")
    return summary


def physics_context_vw_for(variant: str, var_names: list) -> str:
    ctx = PHYSICS_CONTEXT_VW.format(variant=variant)
    if "yw" in var_names:
        ctx += """

壁面距离特征 (本轮新增, 请务必考虑):
- yw = 1-|y/h|: 无量纲真壁距, 0=壁面, 1=槽道中心.
- 动机: Re_τ=180 无对数层, 内外尺度不分离, 壁面约束效应显式依赖离壁距离.
- b23 真值在壁面趋零 (无滑移), 中心最大 — yw 可直接构造壁面阻尼因子
  (如 exp(-yw/τ) 或 1/(1+(yw/τ)^2) 等饱和/衰减结构).
- 初始候选请至少一半包含 yw 项."""
    return ctx


PHYSICS_CONTEXT_VW = """ASM (代数应力模型) 旋转槽道流, Re_τ=180, Ro_τ=7.5/15/30/75/150.
本次任务只拟合雷诺应力的 <vw> 分量 (b23), 即 y-z 平面切应力.
涡量输运方程 (2) 中 <vw> 的二阶横向导数项是核心物理量, 因此模型只需预测
b23(y) 这一个标量剖面, 不需要同时拟合 uu/vv/ww/uv/uw.

输入特征说明:
- variant={variant}: 使用对应张量基变体; m 为 v/g/e 三变体混合 (43 维).
- T1..T10: Pope (1975) 10 个对称无迹张量基在该 (y,z) 分量的取值.
- eta2, zeta2, s3, so2, s2o2, ret, roloc: 局部标量不变量.
目标: b23 = <vw>/(2k) - δ23/3 (因 i≠j, 即 <vw>/(2k)).

关键提示:
1. b23 真值是 y 的偶函数单峰剖面, 中心附近最大, 壁面附近趋零.
2. 只关注 b23 的 NMSE, 不要再优化其他分量.
3. 旋转效应通过 roloc 进入; v/g/e 三种 W 构造分别对应 Wbar, Wbar+Ω, Wbar+2Ω.
4. 推荐使用 tanh 等饱和函数描述壁面约束, 但避免 tanh 大增益导致平台化.
硬规则: 仅允许全局常数 C#, 禁止逐工况参数 K# (ASM 普适性)."""


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v", "g", "e", "m"], required=True)
    ap.add_argument("--method", choices=["pysr", "lgf_go", "lgf_mg"], required=True)
    ap.add_argument("--backend", choices=list(BACKENDS), default="heuristic",
                    help="LGF 后端 (pysr 忽略)")
    ap.add_argument("--budget", type=float, default=600.0, help="LGF 时间预算(s)")
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--sub_sample", type=int, default=300)
    ap.add_argument("--max_edges", type=int, default=16)
    ap.add_argument("--max_params", type=int, default=6)
    ap.add_argument("--pysr_iters", type=int, default=60)
    ap.add_argument("--pysr_maxsize", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--add_yw", action="store_true",
                    help="特征池追加 yw=(1-|yh|)*180 真壁距 (本次低 Re 修正实验)")
    ap.add_argument("--tag", default="", help="run 目录后缀 (如 _yw)")
    ap.add_argument("--yw_scale", type=float, default=1.0,
                    help="yw 尺度: 1.0 => [0,1] 无量纲半槽宽 (默认); "
                         "180.0 => [0,180] 壁面单位 (上一轮口径)")
    args = ap.parse_args()

    data, var_names = load_vw_data(args.variant, add_yw=args.add_yw,
                                   yw_scale=args.yw_scale)
    print(f"[vw-only] variant={args.variant} n_features={len(var_names)} "
          f"cases={[c['Nv'] for c in data]}")

    method_label = args.method
    run_dir = OUT_DIR / f"runs_vwonly{args.tag}_{method_label}" / args.variant

    if args.method == "pysr":
        run_pysr_vw(args.variant, data, var_names,
                    niterations=args.pysr_iters, maxsize=args.pysr_maxsize,
                    populations=8, population_size=40, seed=args.seed,
                    run_dir=run_dir)
    else:
        run_lgf_vw(args.variant, data, var_names,
                   method=args.method, backend=args.backend,
                   budget=args.budget, pop=args.pop, gens=args.gens,
                   sub_sample=args.sub_sample, max_edges=args.max_edges,
                   max_params=args.max_params, seed=args.seed,
                   run_dir=run_dir)


if __name__ == "__main__":
    main()
