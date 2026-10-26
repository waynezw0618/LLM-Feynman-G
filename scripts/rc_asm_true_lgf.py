# -*- coding: utf-8 -*-
"""rc_asm_true_lgf.py — 真 ASM 方法2: LGF (LLM 引导图进化, MiniMax M2.7).

与 PySR 纯 SR 基线 (rc_asm_true_pysr.py) 严格隔离:
  - 本方法走 LGF 完整管线 (G_INIT / M_GUIDE / attribution / GP),
    LLM 用 MiniMax-M2.7, prompt 注入任务物理上下文 (张量基/不变量/
    Re_t/Ro_loc/涡黏参考结构) — "充分发挥 LLM 的物理基础";
  - PySR 不注入任何物理提示, 不借用 LGF 任何策略.

真 ASM 硬约束: global_params_only=True — DSL 只允许全局常数 C#,
含逐工况常数 K# 的候选在三处被拒 (G_INIT 后 / M_GUIDE apply 后 / 精修).

两臂 (用户指定):
  --arm mg : M_GUIDE 全程开启 (每 llm_every 代 LLM 定向变异)
  --arm go : ginit_only — LLM 只出 G_INIT 初始种群, 进化全程离线

数据: outputs/asm/dataset_trueasm_{variant}.npz (17 特征行堆叠, 5 工况).
loss = paramfit mean_c(MSE_c/var_c) = loss_asm 口径 (与 PySR 加权一致).

用法:
  python -X utf8 scripts/rc_asm_true_lgf.py --variant v --arm mg
      [--backend minimax] [--budget 900] [--pop 60] [--gens 40] [--seed 0]
输出: outputs/asm/runs_trueasm_lgf/{variant}_{arm}/
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lgf.config import LGFConfig
from lgf import llm_ops
from lgf import dsl as lgfdsl
from lgf.pipeline_lgf import run_lgf
from rc_asm_run import BACKENDS  # noqa: E402
from rc_asm_true_eval import (load_variant, eval_predictions, summarize,  # noqa: E402
                              lgf_predict)
from rc_asm_true_prep import FEAT_NAMES  # noqa: E402

OUT_DIR = os.path.join(REPO, "outputs", "asm")

# ---- 任务物理上下文 (G_INIT / M_GUIDE prompt 注入) ----
PHYSICS_CONTEXT = """雷诺应力各向异性张量 b_ij = <u_i'u_j'>/(2k) - δ_ij/3 的代数应力模型 (ASM).
流动: 旋转槽道流 DNS, Re_τ=180, Ro_τ=7.5/15/30/75/150 (系统旋转 Ω 沿流向 x1).
特征含义:
- T1..T10: Pope(1975) 十个对称无迹张量基 T^(m)_ij 在当前张量分量 (ij) 处的取值
  (S_ij 应变率与 W_ij 旋转率均已乘 k/eps 归一化). T2∝S_ij(应变), T4∝W²_ij(旋转平方),
  T5∝(SW+WS)_ij(应变-旋转耦合), 其余为高阶组合.
- eta2=tr(S²), zeta2=tr(W²): 二次不变量 (本流场二者数值几乎相等);
  s2o2=tr(S²W²); s3=so2 恒为 0 (二维平均流), 可忽略.
- ret=k²/(ν·eps): 局部湍流雷诺数, 跨约 9 个量级 (槽道中心大, 壁面小).
- roloc=2Ωk/eps: 带符号局部旋转数, 区分 5 个旋转工况的主特征; Ro_τ 越高 roloc 越大,
  高旋转下流动准二维化, Taylor–Görtler 涡卷主导输运.
结构约束 (硬性):
- 目标是 (工况, 壁面位置, 张量分量) 行上的 b_ij 值; 同一公式必须对全部
  6 个分量 (uu,vv,ww,uv,uw,vw) × 5 个工况同时成立.
- 分量方向信息只能经 T1..T10 进入; eta2/ret/roloc 等标量在固定点对 6 个分量
  取值相同, 单独使用它们无法区分分量 (会被行堆叠约束直接惩罚).
- 推荐结构: b_ij = Σ_m g_m(eta2, s2o2, ret, roloc) · T_m_ij.
物理参考:
- 经典线性涡黏 b_ij ≈ -2c_μ(k/eps)S_ij 即 g_2=-c_μ 常数; 该闭包在全部 10 基
  常数版 (cc10) 残差约 0.70, 单常数版更差 — 需要非线性项与旋转调制.
- 高 Ro 极限: 各向异性趋于准二维饱和, g_m 应随 roloc 增大趋于平台
  (tanh(C1*roloc+C2) / 有理函数 C1/(C2+roloc) 类); 低 Ro 近壁 ret 调制湍流衰减.
- 壁面附近 ret 骤降, 可作阻尼因子 (如 ret/(C+ret) 形式).
硬规则: 只允许全局常数 C1,C2,... (跨 5 工况通用); 禁止逐工况常数 K# —
含 K# 的 DSL 会被系统直接拒绝 (违反 ASM 普适性)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v", "g", "e"], required=True)
    ap.add_argument("--arm", choices=["mg", "go"], required=True)
    ap.add_argument("--backend", default="minimax", choices=list(BACKENDS))
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--sub_sample", type=int, default=300)
    ap.add_argument("--max_edges", type=int, default=16)
    ap.add_argument("--max_params", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--llm_api_key", default=None)
    args = ap.parse_args()

    data = load_variant(args.variant)
    cases = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}") for c in data]
    var_names = list(FEAT_NAMES)
    print(f"[lgf-trueasm:{args.variant}/{args.arm}] "
          f"cases={[c['name'] for c in cases]} feats={len(var_names)}")

    be = BACKENDS[args.backend]
    model = args.llm_model or be["model"]
    api_key = (args.llm_api_key or os.environ.get("LGF_LLM_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    client = llm_ops.make_client(be["backend"], url=be["url"], model=model,
                                 api_key=api_key, timeout=240.0)
    if args.backend != "heuristic" and not api_key:
        print(f"[lgf-trueasm][WARN] 后端 {args.backend} 无 API key, "
              f"LGF 会静默退化启发式, 结果不可计入 LLM 后端对比!")
        raise SystemExit(1)

    seed = zlib.crc32(f"trueasm|{args.variant}|{args.arm}".encode()) % (2**31)
    seed = (seed + args.seed) % (2**31)
    cfg = LGFConfig(
        population=args.pop, generations=args.gens,
        sub_sample=args.sub_sample, time_budget_s=args.budget,
        max_edges=args.max_edges, max_params=args.max_params,
        final_starts=8, final_maxiter=150,
        llm_backend=be["backend"], llm_url=be["url"], llm_model=model,
        llm_api_key=api_key, llm_timeout=240.0, seed=seed,
        llm_ginit_only=(args.arm == "go"),
        wall_seeds=False,                       # 无 yw 特征, 显式关闭
        physics_context=PHYSICS_CONTEXT,        # true-ASM 物理上下文
        global_params_only=True,                # 真 ASM: 禁 K# 逐工况常数
        ginit_retry=2, ginit_min_kept=3, ginit_fewshot=True,
    )

    run_dir = os.path.join(OUT_DIR, "runs_trueasm_lgf", f"{args.variant}_{args.arm}")
    os.makedirs(run_dir, exist_ok=True)
    name = f"trueasm_{args.variant}_{args.arm}"
    t0 = time.time()
    res = run_lgf(cases, var_names, cfg, run_dir, client, name=name)
    elapsed = time.time() - t0

    # ---- 统一评估: best + top_final 逐条 (loss_asm + rho_a) ----
    entries = [("best", res["best"])] + [
        (f"top{i}", e) for i, e in enumerate(res.get("top_final", []))]
    evaluated = []
    for tag, e in entries:
        rec = dict(tag=tag, dsl=e["dsl"], loss_paramfit=e["loss"],
                   complexity=e["complexity"], theta=e["theta"],
                   n_g=len(lgfdsl.parse_dsl(e["dsl"], var_names).global_names()),
                   n_c=len(lgfdsl.parse_dsl(e["dsl"], var_names).case_names()))
        try:
            preds, _ = lgf_predict(e["dsl"], e["theta"], data)
            rec["metrics"] = eval_predictions(preds, data)
            if tag == "best":
                preds_file = os.path.join(run_dir, "preds_best.npz")
                np.savez_compressed(preds_file,
                                    **{f"c{i}": p for i, p in enumerate(preds)})
                rec["preds_file"] = preds_file
        except Exception as ex:
            rec["metrics"] = dict(error=f"{type(ex).__name__}: {ex}")
        evaluated.append(rec)
        m = rec["metrics"]
        print(f"[lgf-trueasm:{args.variant}/{args.arm}] {tag}: "
              f"cplx={rec['complexity']} loss={e['loss']:.5f} :: "
              f"{summarize(m) if m.get('per_case') else m.get('error')}")

    summary = dict(
        variant=args.variant, arm=args.arm, method="lgf",
        backend=args.backend, model=model, seed=seed,
        config=dict(budget=args.budget, pop=args.pop, gens=args.gens,
                    sub_sample=args.sub_sample, max_edges=args.max_edges,
                    max_params=args.max_params,
                    llm_ginit_only=(args.arm == "go"),
                    global_params_only=True, physics_context=PHYSICS_CONTEXT),
        feat_names=var_names, elapsed_s=round(elapsed, 1),
        llm_stats=res.get("llm_stats", {}),
        entries=evaluated)
    out = os.path.join(run_dir, "result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[lgf-trueasm:{args.variant}/{args.arm}] -> {out}")


if __name__ == "__main__":
    main()
