# -*- coding: utf-8 -*-
"""rc_asm_vw_t23_full.py — vw(b23) + T₂₃ 张量通道全量轮: PySR vs LGF × {v,g,e,m}.

目的 (用户 2026-10-12 指令): 把对称性允许的 4 个非零 T⁽ᵐ⁾₂₃ 通道 (RMS 归一)
加入特征池, 用 PySR 与 LGF (go/mg 两臂, glm53) 同台竞争, 找 LGF 优的证据.

特征池 (与 10-12 夜 rc_asm_vw_t23_pysr.py 的 t23 池同构, 扩展到 m):
  v/g/e: [eta2,zeta2,s3,so2,s2o2,ret,roloc, yw] + [T1s,T5s,T6s,T10s]   = 12 维
         (T⁽ᵐ⁾s 取自该变体基; 通道跨 5 工况 pooled RMS 归一)
  m    : dataset_trueasm_mix 43 维 + yw + [T1s, T5s/T6s/T10s × v/g/e]  = 54 维
口径: b23 单分量, yw∈[0,1], NMSE 分母=逐工况 var_c(b23), 权重 1/(Nv·var_c).
LGF: physics context 注入 T23 通道物理 + 张量基 ASM 理论结构 (a_m(I)·T⁽ᵐ⁾);
     PySR 盲跑不注入 (与 V4b 方法学一致).

输出: outputs/asm/runs_vw_t23full_{pysr,lgf_go,lgf_mg}/{variant}/result.json

用法:
  python -X utf8 scripts/rc_asm_vw_t23_full.py --variant v --method pysr
  python -X utf8 scripts/rc_asm_vw_t23_full.py --variant g --method lgf_mg --backend glm53
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

from rc_asm_vw_only import (  # noqa: E402
    eval_vw_predictions, summarize_vw, lgf_predict_vw,
    extract_vw_only, run_pysr_vw)
from rc_asm_true_mix import load_mix, build_mix_feat_names  # noqa: E402
from rc_asm_run import BACKENDS  # noqa: E402
from lgf.config import LGFConfig  # noqa: E402
from lgf import llm_ops, dsl as lgfdsl  # noqa: E402
from lgf.pipeline_lgf import run_lgf  # noqa: E402

OUT = REPO / "outputs" / "asm"
ROS = [7.5, 15, 30, 75, 150]
IDX23 = 5
CHANS = ["T1", "T5", "T6", "T10"]
SCAL = ["eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc"]


# ----------------------------------------------------------------------------
# 数据 / 特征池
# ----------------------------------------------------------------------------
def _rms_norm(cases_chan: list) -> list:
    """跨 5 工况 pooled RMS 归一 (每通道除以 pooled RMS)."""
    pooled = np.vstack(cases_chan)
    rms = np.sqrt(np.mean(pooled ** 2, axis=0))
    rms[rms <= 0] = 1.0
    return [c / rms for c in cases_chan]


def load_t23_pool(variant: str):
    """返回 (cases, var_names). cases[i]['Xfit'] = 特征矩阵 (Nv, n_feat)."""
    if variant == "m":
        base = extract_vw_only(load_mix())
        names = build_mix_feat_names() + ["yw"]
        # 通道: T1 共享 + T5/T6/T10 × {v,g,e}
        dss = {v: np.load(OUT / f"dataset_trueasm_{v}.npz") for v in ("v", "g", "e")}
        chan_names = ["T1s"]
        for t in ("T5", "T6", "T10"):
            for v in ("v", "g", "e"):
                chan_names.append(f"{t}s_{v}")
        raw = []
        for i in range(len(base)):
            p = f"c{i}_"
            cols = []
            for t in ("T1", "T5", "T6", "T10"):
                srcs = ("v",) if t == "T1" else ("v", "g", "e")
                for v in srcs:
                    m = np.asarray(dss[v][p + "valid"], bool)
                    cols.append(dss[v][p + t][m, IDX23])
            raw.append(np.column_stack(cols))
        normed = _rms_norm(raw)
        for i, c in enumerate(base):
            yw = 1.0 - np.abs(c["yh"])
            c["Xfit"] = np.column_stack([c["Xfit"], yw, normed[i]])
        return base, names + chan_names

    ds = np.load(OUT / f"dataset_trueasm_{variant}.npz")
    cases, raw = [], []
    for i, ro in enumerate(ROS):
        p = f"c{i}_"
        valid = ds[p + "valid"].astype(bool)
        yh = ds[p + "yh"][valid]
        scal = ds[p + "scalars"][valid]
        yw = 1.0 - np.abs(yh)
        b23 = ds[p + "b6"][valid, IDX23].astype(float)
        chans = np.column_stack([ds[p + m][valid, IDX23] for m in CHANS])
        raw.append(chans)
        cases.append(dict(i=i, ro=ro, yh=yh, b23=b23,
                          scal=scal, yw=yw, Nv=int(valid.sum()),
                          var_c=float(np.var(b23)),
                          max_abs_ref=float(np.max(np.abs(b23)))))
    normed = _rms_norm(raw)
    for c, ch in zip(cases, normed):
        c["Xfit"] = np.column_stack([c["scal"], c["yw"], ch])
        c["yfit"] = c["b23"]
    names = SCAL + ["yw"] + [f"{m}s" for m in CHANS]
    return cases, names


# ----------------------------------------------------------------------------
# LGF
# ----------------------------------------------------------------------------
PHYSICS_T23_HEAD = """ASM (代数应力模型) 旋转槽道流 DNS, Re_τ=180, Ro_τ=7.5/15/30/75/150 (系统旋转 Ω 沿流向).
本次任务只拟合雷诺应力的 <vw> 分量 (b23 = <vw>/(2k), y-z 平面切应力). b23 真值剖面:
全槽道恒正、偶对称单峰, 槽道中心附近最大, 壁面趋零 (无滑移).

特征池 = 局部标量不变量 + 真壁距 + 张量基 T⁽ᵐ⁾₂₃ 通道 (跨工况 RMS 归一):
- eta2=tr(S²), zeta2=tr(W²); s3/so2 恒为 0 (二维平均流), 可忽略; s2o2=tr(S²W²);
  ret=k²/(ν·ε) 局部湍流雷诺数; roloc=2Ωk/ε 局部旋转数 — 区分 5 个旋转工况的核心
  变量 (Ro_τ 越大 roloc 越大, 高旋转下流动准二维化);
  yw=1-|y/h|∈[0,1]: 0=壁面, 1=槽道中心.
- T1s = S₂₃ = ½·(∂w̄/∂y) (乘 k/ε 归一): 展向平均剪切生成项. **实测与 b23 同奇偶 —
  关于槽道中心偶对称** (偶残差 0.25%), 两壁为零; 切勿按反对称处理. 幅值 O(1),
  但剖面随 y 起伏明显, 单靠 T1s 无法复现 b23 的单峰.{chan_note}

结构线索 (张量基 ASM 理论, 强烈建议参考):
1. b_ij 的闭包原生形式是 b_ij = Σ_m a_m(不变量)·T⁽ᵐ⁾_ij. 对 b23 即
   b23 ≈ a₁(roloc,不变量)·T1s + a₅(...)·T5s + a₆(...)·T6s + a₁₀(...)·T10s.
   四个 T 通道与 b23 同奇偶 (均为偶), 可直接线性组合.
   **硬约束: 系数必须含 roloc (或等价不变量) 依赖 — 常数系数跨工况 NMSE≈3-5 崩溃;
   系数只是 roloc 的线性函数也只到 ~0.5-0.7. 系数须是足够丰富的 roloc/不变量函数.**
2. Ro 标度: 中低旋转下 b23 随 √roloc 增长 (科氏生产); 旋转很强时准二维化,
   b23 应饱和或回落 — tanh(C·roloc)、有理函数 C/(C'+roloc)、√roloc·(1-C''·roloc)
   均可描述该非单调行为.
3. 近壁: 结合 yw 的对数律型或饱和型因子 (单 log(yw) 壁面发散, 须配饱和/旋转因子抑制).
{scale_note}硬规则: 只允许全局常数 C# (5 工况通用), 禁止逐工况常数 K# (ASM 普适性)."""

_CHAN_NOTE = {
    "v": "\n- T5s/T6s/T10s: v 基 (W=W̄ 纯涡量) 高阶通道, 不含 Ω, 跨工况动态范围温和, 可直接使用.",
    "g": ("\n- T5s/T6s/T10s: g 基 (W=W̄+Ω) 把 Ω 注入 W — T5s/T10s 含 Ω 高次幂, 跨工况动态范围"
          "差 2-3 个量级, 直接用大系数必失败; 只可用比值 (T5s/(C#+T10s), T6s·T1s/roloc) 或饱和抑制."),
    "e": ("\n- T5s/T6s/T10s: e 基 (W=W̄+2Ω 扩展内禀旋转) 把 2Ω 注入 W — T5s/T10s 含 Ω 高次幂, "
          "跨工况动态范围差 2-3 个量级, 直接用大系数必失败; 只可用比值 (T5s/(C#+T10s), "
          "T6s·T1s/roloc) 或饱和抑制."),
    "m": ("\n- 本池为 v/g/e 43 维混合池 + 10 个归一 T23 通道 (T1s 共享, T5s/T6s/T10s × v/g/e). "
          "同名通道三变体差异仅在是否注入 Ω: v 系不含 Ω 动态范围温和, g/e 系含 Ω 高次幂跨工况 "
          "2-3 个量级 — 优先用 v 系通道或比值结构."),
}

_SCALE_NOTE = ("- 尺度病警告: T5s/T10s 的裸比值/单项式在高 Ro 工况会爆值, 须配比值/1/roloc/tanh "
               "饱和等组合抑制.\n")


def physics_t23_for(variant: str) -> str:
    return PHYSICS_T23_HEAD.format(chan_note=_CHAN_NOTE[variant],
                                   scale_note=_SCALE_NOTE if variant != "v" else "")


def run_lgf_t23(variant: str, data: list, var_names: list, *, method: str,
                backend: str, budget: float = 600.0, pop: int = 60,
                gens: int = 40, sub_sample: int = 300, max_edges: int = 16,
                max_params: int = 6, seed: int = 0, run_dir: Path = None):
    be = BACKENDS[backend]
    api_key = os.environ.get("LGF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if backend != "heuristic" and not api_key:
        raise SystemExit(f"[t23-lgf:{variant}] backend={backend} 需 LGF_LLM_API_KEY")
    client = llm_ops.make_client(be["backend"], url=be["url"], model=be["model"],
                                 api_key=api_key, timeout=300.0,
                                 max_tokens=be.get("max_tokens", 8192),
                                 extra_body=be.get("extra_body"))

    cases = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}") for c in data]
    print(f"[t23-lgf:{variant}] cases={[c['name'] for c in cases]} "
          f"feats={len(var_names)} method={method} backend={backend}", flush=True)

    ginit_only = (method == "lgf_go")
    s = (zlib.crc32(f"vw_t23full|{variant}|{method}".encode()) + seed) % (2 ** 31)
    cfg = LGFConfig(
        population=pop, generations=gens, sub_sample=sub_sample,
        time_budget_s=budget, max_edges=max_edges, max_params=max_params,
        final_starts=8, final_maxiter=150,
        llm_backend=be["backend"], llm_url=be["url"], llm_model=be["model"],
        llm_api_key=api_key, llm_timeout=240.0, seed=s,
        llm_ginit_only=ginit_only, wall_seeds=False,
        physics_context=physics_t23_for(variant),
        global_params_only=True,
        ginit_retry=2, ginit_min_kept=3, ginit_fewshot=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"vw_t23full_{variant}_{method}"
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
            preds = lgf_predict_vw(e["dsl"], e["theta"], data, var_names)
            rec["metrics"] = eval_vw_predictions(preds, data)
            if tag == "best":
                preds_file = run_dir / "preds_best.npz"
                np.savez_compressed(preds_file,
                                    **{f"c{i}": p for i, p in enumerate(preds)})
                rec["preds_file"] = str(preds_file)
        except Exception as ex:
            rec["metrics"] = dict(error=f"{type(ex).__name__}: {ex}")
        evaluated.append(rec)
        m = rec["metrics"]
        print(f"[t23-lgf:{variant}] {tag}: cplx={rec['complexity']} "
              f"loss={e['loss']:.5f} :: "
              f"{summarize_vw(m) if m.get('per_case') else m.get('error')}", flush=True)

    summary = dict(
        variant=variant, method=method, target="vw", target_label="b23/<vw>",
        pool="t23", backend=backend, model=be["model"], seed=s,
        config=dict(budget=budget, pop=pop, gens=gens, sub_sample=sub_sample,
                    max_edges=max_edges, max_params=max_params,
                    llm_ginit_only=ginit_only, global_params_only=True,
                    n_features=len(var_names)),
        feat_names=list(var_names), elapsed_s=round(elapsed, 1),
        llm_stats=res.get("llm_stats", {}), entries=evaluated)
    out = run_dir / "result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[t23-lgf:{variant}] -> {out}", flush=True)
    return summary


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v", "g", "e", "m"], required=True)
    ap.add_argument("--method", choices=["pysr", "lgf_go", "lgf_mg"], required=True)
    ap.add_argument("--backend", choices=list(BACKENDS), default="glm53")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--sub_sample", type=int, default=300)
    ap.add_argument("--max_edges", type=int, default=16)
    ap.add_argument("--max_params", type=int, default=6)
    ap.add_argument("--pysr_iters", type=int, default=60)
    ap.add_argument("--pysr_maxsize", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="输出目录后缀 (如 _ctl); 空=原目录")
    args = ap.parse_args()

    data, var_names = load_t23_pool(args.variant)
    print(f"[t23-full] variant={args.variant} n_features={len(var_names)} "
          f"rows/case={[c['Nv'] for c in data]} tag='{args.tag}'", flush=True)

    run_dir = OUT / f"runs_vw_t23full_{args.method}{args.tag}" / args.variant
    if args.method == "pysr" and args.seed != 0:
        run_dir = OUT / f"runs_vw_t23full_{args.method}{args.tag}" / f"{args.variant}_s{args.seed}"
    if run_dir.exists() and (run_dir / "result.json").exists():
        print(f"[t23-full] SKIP: {run_dir/'result.json'} 已存在", flush=True)
        return

    if args.method == "pysr":
        run_pysr_vw(args.variant, data, var_names,
                    niterations=args.pysr_iters, maxsize=args.pysr_maxsize,
                    populations=8, population_size=40, seed=args.seed,
                    run_dir=run_dir)
    else:
        run_lgf_t23(args.variant, data, var_names, method=args.method,
                    backend=args.backend, budget=args.budget, pop=args.pop,
                    gens=args.gens, sub_sample=args.sub_sample,
                    max_edges=args.max_edges, max_params=args.max_params,
                    seed=args.seed, run_dir=run_dir)


if __name__ == "__main__":
    main()
