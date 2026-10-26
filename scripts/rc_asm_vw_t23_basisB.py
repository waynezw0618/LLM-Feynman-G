# -*- coding: utf-8 -*-
"""rc_asm_vw_t23_basisB.py — 方案B: 纯张量基口径的 b23 闭合.

模型形式 (用户 2026-10-16 指令):
    b23(y) = Σ_{m∈{1,5,6,10}} a_m(Θ) · T̃⁽ᵐ⁾₂₃(y),
    Θ = Pope 不变量 (I1=eta2, I2=zeta2, I4=s2o2) + roloc + ret (+ yw 壁面扩展),
    T̃ = T 通道按跨工况 pooled RMS 归一.

两阶段:
  stage base — 结构化加权 lstsq 定系数 (rich 基, 全局常数) 得到
               (i) 结构化基线精度, (ii) 每条系数 a_m 在全部点上的靶标值;
  stage sr   — LGF (lgf_go/lgf_mg) / PySR 对每条 a_m(Θ) 做符号回归,
               再用发现的闭式重组 b23 并评估 (同一 NMSE 口径).

评估口径同 rc_asm_vw_only.eval_vw_predictions:
  loss_vw = mean_c[ mean_y (b23_pred-b23)² / var_c(b23) ].

用法:
  python -X utf8 scripts/rc_asm_vw_t23_basisB.py --stage base --variant g
  python -X utf8 scripts/rc_asm_vw_t23_basisB.py --stage sr --variant g --method pysr
  python -X utf8 scripts/rc_asm_vw_t23_basisB.py --stage sr --variant g --method lgf_go --backend glm53
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rc_asm_run import BACKENDS  # noqa: E402
from lgf.config import LGFConfig  # noqa: E402
from lgf import llm_ops  # noqa: E402
from lgf.pipeline_lgf import run_lgf  # noqa: E402
from lgf.compiler import CompiledGraph  # noqa: E402
from lgf import dsl as lgfdsl  # noqa: E402

OUT = REPO / "outputs" / "asm"
ROS = [7.5, 15, 30, 75, 150]
IDX23 = 5
CHANS = [1, 5, 6, 10]
BASE = ["eta2", "zeta2", "s2o2", "ret", "roloc", "yw"]

# 系数基 (rich pool C): Pope 不变量 + roloc + ret + 壁面 yw
POOL = [
    ("1", lambda f: np.ones_like(f["yw"])),
    ("sqrt(roloc)", lambda f: np.sqrt(f["roloc"])),
    ("roloc", lambda f: f["roloc"]),
    ("tanh(roloc)", lambda f: np.tanh(f["roloc"])),
    ("eta2", lambda f: f["eta2"]),
    ("zeta2", lambda f: f["zeta2"]),
    ("s2o2", lambda f: f["s2o2"]),
    ("ret", lambda f: f["ret"]),
    ("log(eta2)", lambda f: np.log(f["eta2"] + 1e-6)),
    ("yw", lambda f: f["yw"]),
    ("yw^2", lambda f: f["yw"] ** 2),
    ("yw*sqrt(roloc)", lambda f: f["yw"] * np.sqrt(f["roloc"])),
]


# ---------------------------------------------------------------------------
def load_cases(variant: str):
    ds = np.load(OUT / f"dataset_trueasm_{variant}.npz")
    cases = []
    for i, ro in enumerate(ROS):
        p = f"c{i}_"
        v = ds[p + "valid"].astype(bool)
        yh = ds[p + "yh"][v].astype(float)
        s = ds[p + "scalars"][v].astype(float)
        b23 = ds[p + "b6"][v, IDX23].astype(float)
        ch = {m: ds[p + f"T{m}"][v, IDX23].astype(float) for m in CHANS}
        f = dict(eta2=s[:, 0], zeta2=s[:, 1], s2o2=s[:, 4], ret=s[:, 5],
                 roloc=s[:, 6], yw=1.0 - np.abs(yh))
        cases.append(dict(ro=ro, yh=yh, b23=b23, f=f, ch=ch, Nv=len(yh),
                          var_c=float(np.var(b23))))
    return cases


def rms_per_channel(cases):
    r = {}
    for m in CHANS:
        pooled = np.concatenate([c["ch"][m] for c in cases])
        r[m] = float(np.sqrt(np.mean(pooled ** 2))) or 1.0
    return r


def design(cases, chans, pool):
    """列 = φ_k(f)·T̃_m23, 按通道分块."""
    Z = []
    for c in cases:
        cols, names = [], []
        for m in chans:
            t = c["ch"][m] / c["_rms"][m]
            for kn, fn in pool:
                cols.append(fn(c["f"]) * t)
                names.append(f"{kn}*T{m}t")
        Z.append(np.column_stack(cols))
    return Z, names


def wlstsq(Z, y, w):
    sw = np.sqrt(w)
    Zs, ys = Z * sw[:, None], y * sw
    sc = np.sqrt((Zs ** 2).sum(0)); sc[sc <= 0] = 1.0
    cs, *_ = np.linalg.lstsq(Zs / sc, ys, rcond=1e-12)
    return cs / sc


def eval_nmse(pred_list, cases):
    pc, nms = {}, []
    for p, c in zip(pred_list, cases):
        r = float(np.mean((np.asarray(p) - c["b23"]) ** 2) / c["var_c"])
        nms.append(r); pc[f"Ro{c['ro']:g}"] = round(r, 4)
    return dict(loss_vw=float(np.mean(nms)), worst=float(np.max(nms)), per_case=pc)


def weight(cases):
    return np.concatenate([np.full(c["Nv"], 1.0 / (c["Nv"] * c["var_c"]))
                           for c in cases])


# ---------------------------------------------------------------------------
# stage base
# ---------------------------------------------------------------------------
def stage_base(variant):
    cases = load_cases(variant)
    rms = rms_per_channel(cases)
    for c in cases:
        c["_rms"] = rms
    Z, names = design(cases, CHANS, POOL)
    Zall = np.vstack(Z)
    yall = np.concatenate([c["b23"] for c in cases])
    wall = weight(cases)
    coef = wlstsq(Zall, yall, wall)
    preds, off = [], 0
    for c in cases:
        preds.append(Zall[off:off + c["Nv"]] @ coef); off += c["Nv"]
    r = eval_nmse(preds, cases)

    # 每条系数在全部点上的靶标值
    nb = len(POOL)
    tgt = {}
    for j, m in enumerate(CHANS):
        seg = coef[j * nb:(j + 1) * nb]
        tgt[m] = [np.column_stack([fn(c["f"]) for _, fn in POOL]) @ seg for c in cases]

    save = {"variant": variant, "chans": np.array(CHANS),
            "base": np.array(BASE), "pool": np.array([n for n, _ in POOL]),
            "rms": np.array([rms[m] for m in CHANS]), "coef": coef}
    for i, c in enumerate(cases):
        save[f"c{i}_X"] = np.column_stack([c["f"][k] for k in BASE])
        save[f"c{i}_b23"] = c["b23"]
        save[f"c{i}_ro"] = c["ro"]
        save[f"c{i}_var_c"] = c["var_c"]
        for m in CHANS:
            save[f"c{i}_a{m}"] = tgt[m][i]
            save[f"c{i}_T{m}t"] = c["ch"][m] / rms[m]
    np.savez_compressed(OUT / f"basisB_targets_{variant}.npz", **save)

    summary = dict(variant=variant, stage="base", pool=[n for n, _ in POOL],
                   rms=rms, loss_vw=r["loss_vw"], worst=r["worst"],
                   per_case=r["per_case"], n_params=len(coef),
                   coef=[[float(v) for v in coef[j * nb:(j + 1) * nb]]
                         for j in range(len(CHANS))])
    print(f"[basisB:base:{variant}] 结构化纯张量基: loss={r['loss_vw']:.4f} "
          f"worst={r['worst']:.4f} | " +
          " ".join(f"{k[2:]}:{v:.3f}" for k, v in r["per_case"].items()), flush=True)
    print("-- 结构化系数 (a_m = Σ c_k·φ_k) --", flush=True)
    for j, m in enumerate(CHANS):
        seg = coef[j * nb:(j + 1) * nb]
        print(f"   a{m:<2}(与 T̃{m}t 配对) = " + " + ".join(
            f"({v:+.4g})·{k}" for v, k in zip(seg, [n for n, _ in POOL])), flush=True)
    for i, c in enumerate(cases):
        print(f"   a_m 靶标 Ro{c['ro']:g}: " + " ".join(
            f"a{m}∈[{tgt[m][i].min():+.3g},{tgt[m][i].max():+.3g}]" for m in CHANS),
            flush=True)
    json.dump(summary, open(OUT / f"basisB_base_{variant}.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    return summary


# ---------------------------------------------------------------------------
# stage sr
# ---------------------------------------------------------------------------
def _cases_for_target(variant, m):
    """系数回归的案例集.

    口径修正 (2026-10-16): 目标是让**重组 b23 的误差**最小, 而不是让系数自身
    误差最小. b23 = Σ_m a_m·T̃_m, 故逐点权重取 w_i = T̃_m,i² / (Nv·var_c(b23)),
    此时 Σ_i w_i·(Δa)² 恰为该通道对 b23 的 (逐工况 NMSE) 贡献.
    LGF 的 compiler 对 case["w"] 正是这个语义 (见 lgf/compiler.py:104-107).
    """
    d = np.load(OUT / f"basisB_targets_{variant}.npz")
    pooled = np.concatenate([d[f"c{i}_a{m}"].astype(float) for i in range(len(ROS))])
    scale = float(np.sqrt(np.mean(pooled ** 2))) or 1.0
    cs = []
    for i in range(len(ROS)):
        X = d[f"c{i}_X"].astype(float)
        a = d[f"c{i}_a{m}"].astype(float) / scale
        Tt = d[f"c{i}_T{m}t"].astype(float)
        b23 = d[f"c{i}_b23"].astype(float)
        var_c = float(np.var(b23)) + 1e-300
        nv = X.shape[0]
        w = Tt ** 2 / (nv * var_c)
        cs.append(dict(i=i, ro=float(d[f"c{i}_ro"]), Xfit=X, yfit=a, w=w,
                       scale=scale, Nv=nv, var_c=float(np.var(a))))
    return cs, d


PHYS = """旋转槽道流 ASM 张量基闭合, Re_τ=180, Ro_τ=7.5/15/30/75/150 (系统旋转沿流向).
任务: 拟合 b23 = Σ_m a_m(Θ)·T̃⁽ᵐ⁾₂₃ 中的**张量基系数 a_m(Θ)** (m 依次为 1,5,6,10),
T̃ 为该 T 通道 23 分量按跨工况 RMS 归一后的值 (常数已知, 无需拟合).
本次目标是 **a_{M}**, 即张量通道 {M} 的系数, 已按自己的 RMS 归一为 O(1) 量级.

Θ (系数自变量, 均为逐点局部量):
- eta2 = tr(S²) ≥ 0     (Pope I1; 跨 5 工况 1e-6 → 3e2, 近壁极小、对数变化)
- zeta2 = tr(W²) ≥ 0    (Pope I2; 同量级; 对 g/e 基含 Ω 贡献, 是旋转信息的主要载体之一)
- s2o2 = tr(S²W²) ≤ 0   (Pope 不变量之一; 2e4 量级)
- ret  = k²/(νε) 局部湍流雷诺数 (6e-5 → 5e4)
- roloc = 2Ωk/ε 局部旋转数 (6e-5 → 6.8 / 15.4; 区分 5 个旋转工况的核心变量)
- yw = 1-|y/h| ∈ [0,1] 真壁距 (0=壁面, 1=槽道中心) — 非局部量, 作为壁面扩展提供

先验 (请据此构造候选):
1. 系数是 Θ 的**光滑函数**; 常数项不足 (常数系数跨工况 NMSE 3-5 崩溃).
2. 高旋转 (roloc 大) 下流动准二维化, 系数常出现 tanh(C·roloc)、C/(C'+roloc)、
   exp(-C·roloc)、√roloc 这类饱和/衰减结构.
3. 近壁 eta2/ret 急剧变化 (1e-6 → 1e2), log(eta2+eps)、1/√ret、eta2/(C+eta2)
   一类的比值/对数结构常见; 直接用 eta2 的大幂会爆.
4. 硬规则: 只用全局常数 C# (5 工况通用), 禁止逐工况常数 K#."""


def _fit_pysr(cs, seed, niter, maxsize, run_dir, label):
    from pysr import PySRRegressor
    X = np.vstack([c["Xfit"] for c in cs])
    y = np.concatenate([c["yfit"] for c in cs])
    w = np.concatenate([c["w"] for c in cs])
    w = w / max(w.sum(), 1e-300) * len(w)
    unary = ["exp", "log", "sqrt", "tanh"]
    nested = {u: {v: 0 for v in unary} for u in unary}
    mdl = PySRRegressor(
        niterations=niter, populations=8, population_size=40, maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"], unary_operators=unary,
        nested_constraints=nested, constraints={u: 12 for u in unary},
        model_selection="best", random_state=seed, temp_equation_file=True,
        verbosity=0, progress=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mdl.fit(X, y, weights=w, variable_names=BASE)
    el = time.time() - t0
    preds = [np.asarray(mdl.predict(c["Xfit"]), dtype=float) for c in cs]
    eqs = mdl.equations_
    i = int(np.argmin(eqs["loss"].to_numpy()))
    return dict(dsl=str(eqs.iloc[i]["equation"]),
                complexity=int(eqs.iloc[i]["complexity"]),
                loss=float(eqs.iloc[i]["loss"]), elapsed_s=round(el, 1),
                preds=preds)


def _fit_lgf(cs, method, backend, seed, budget, run_dir, label, mstr):
    be = BACKENDS[backend]
    api_key = os.environ.get("LGF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if backend != "heuristic" and not api_key:
        raise SystemExit(f"[basisB] backend={backend} 需 LGF_LLM_API_KEY")
    client = llm_ops.make_client(be["backend"], url=be["url"], model=be["model"],
                                 api_key=api_key, timeout=300.0,
                                 max_tokens=be.get("max_tokens", 8192),
                                 extra_body=be.get("extra_body"))
    s = (zlib.crc32(f"basisB|{label}".encode()) + seed) % (2 ** 31)
    cfg = LGFConfig(
        population=60, generations=30, sub_sample=300, time_budget_s=budget,
        max_edges=15, max_params=6, final_starts=8, final_maxiter=150,
        llm_backend=be["backend"], llm_url=be["url"], llm_model=be["model"],
        llm_api_key=api_key, llm_timeout=240.0, seed=s,
        llm_ginit_only=(method == "lgf_go"), wall_seeds=False,
        physics_context=PHYS.replace("{M}", mstr),
        global_params_only=True, ginit_retry=2, ginit_min_kept=3,
        ginit_fewshot=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    lgf_cases = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}", w=c["w"])
                 for c in cs]
    t0 = time.time()
    res = run_lgf(lgf_cases, BASE, cfg, str(run_dir), client,
                  name=f"basisB_{label}_{method}")
    el = time.time() - t0
    best = res["best"]
    g = lgfdsl.parse_dsl(best["dsl"], BASE)
    cg = CompiledGraph(g, backend="numpy")
    th = np.asarray(best["theta"], dtype=float)
    preds = [np.asarray(cg.eval_case(th, c["Xfit"]), dtype=float) for c in cs]
    return dict(dsl=best["dsl"], complexity=best["complexity"],
                loss=float(best["loss"]), theta=[float(v) for v in th],
                elapsed_s=round(el, 1), preds=preds)


def stage_sr(variant, method, backend, seed, budget, niter, maxsize):
    cases = load_cases(variant)
    rms = rms_per_channel(cases)
    rows, per_target = [], {}
    for m in CHANS:
        cs, d = _cases_for_target(variant, m)
        run_dir = OUT / f"runs_basisB_{method}{'' if method!='pysr' or seed==0 else f'_s{seed}'}" / f"{variant}_a{m}"
        tag = f"{variant}_a{m}"
        if method == "pysr":
            fit = _fit_pysr(cs, seed, niter, maxsize, run_dir, tag)
        else:
            fit = _fit_lgf(cs, method, backend, seed, budget, run_dir, tag, str(m))
        fit["scale"] = cs[0]["scale"]
        per_target[m] = fit
        print(f"[basisB:sr:{variant}:{method}] a{m}: cplx={fit['complexity']} "
              f"loss_coef={fit['loss']:.3e} {fit['elapsed_s']}s :: {fit['dsl']}",
              flush=True)
        rows.append(dict(target=f"a{m}", dsl=fit["dsl"], complexity=fit["complexity"],
                         loss_coef=fit["loss"], scale=[c["scale"] for c in cs][0],
                         theta=fit.get("theta")))

    # 重组 b23 = Σ_m (scale_m · sr_m) · T̃_m23
    preds = []
    for i in range(len(cases)):
        acc = np.zeros(cases[i]["Nv"])
        for m in CHANS:
            acc = acc + per_target[m]["preds"][i] * per_target[m]["scale"] * \
                (cases[i]["ch"][m] / rms[m])
        preds.append(acc)
    r = eval_nmse(preds, cases)
    print(f"[basisB:sr:{variant}:{method}] 重组 b23: loss={r['loss_vw']:.4f} "
          f"worst={r['worst']:.4f} | " +
          " ".join(f"{k[2:]}:{v:.3f}" for k, v in r["per_case"].items()), flush=True)

    sfx = f"_s{seed}" if (method == "pysr" and seed) else ""
    run_dir = OUT / "runs_basisB" / f"{variant}_{method}{sfx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(variant=variant, method=method, backend=backend, stage="sr",
                   rms=rms, loss_vw=r["loss_vw"], worst=r["worst"],
                   per_case=r["per_case"], targets=rows, seed=seed)
    json.dump(summary, open(run_dir / "result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[basisB:sr:{variant}:{method}] -> {run_dir/'result.json'}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "sr"], required=True)
    ap.add_argument("--variant", choices=["v", "g", "e"], required=True)
    ap.add_argument("--method", choices=["pysr", "lgf_go", "lgf_mg"], default="pysr")
    ap.add_argument("--backend", choices=list(BACKENDS), default="glm53")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--pysr_iters", type=int, default=80)
    ap.add_argument("--pysr_maxsize", type=int, default=25)
    a = ap.parse_args()
    if a.stage == "base":
        stage_base(a.variant)
    else:
        stage_sr(a.variant, a.method, a.backend, a.seed, a.budget,
                 a.pysr_iters, a.pysr_maxsize)


if __name__ == "__main__":
    main()
