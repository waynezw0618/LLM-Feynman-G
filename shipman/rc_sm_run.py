"""shipman.rc_sm_run — 阶段2: 单 run LGF 跑批 (断点续跑).

用法:
  python -X utf8 shipman/rc_sm_run.py --variant v1 --eq Y --noise 0.0 --seed 0 \
      --arm gponly --backend heuristic --budget 300

输出: outputs/shipman/runs_sm/{variant}_{eq}_n{noise}_s{seed}_{arm}/
  result.json (存在即 SKIP) + run_lgf 落盘三件套.
数据: 现场重仿真 (与 prep 同 seed 确定性一致), 避免读盘耦合.
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
sys.path.insert(0, REPO)

from lgf.config import LGFConfig                    # noqa: E402
from lgf.pipeline_lgf import run_lgf                # noqa: E402
from lgf.llm_ops import make_client                 # noqa: E402
from shipman import templates_mmg                   # noqa: E402  (import 即注册模板)
from shipman.coeff_kvlcc2 import kvlcc2             # noqa: E402
from shipman.context import BACKENDS, PHYSICS_CTX_MMG  # noqa: E402
from shipman.variants import build_cases, calibrate_ext  # noqa: E402

OUT_DIR = os.path.join(REPO, "outputs", "shipman", "runs_sm")
MANEUVERS = ("turn35p", "turn35s", "zig1010", "zig2020")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--eq", default="Y", choices=["X", "Y", "N"])
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="gponly", choices=["gponly", "llm"])
    ap.add_argument("--backend", default="heuristic")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--cal_frac", type=float, default=0.10)
    ap.add_argument("--data", default="auto", choices=["auto", "v1", "ext"],
                    help="auto: v1变体→纯v1数据, 其余→扩展数据; v1/ext 强制指定 (误吸收检验用)")
    args = ap.parse_args()

    tag = "" if args.data == "auto" else f"{args.variant}pool_{args.data}data_"
    run_dir = os.path.join(OUT_DIR, f"{tag}{args.variant}_{args.eq}_n{args.noise}_s{args.seed}_{args.arm}")
    result_path = os.path.join(run_dir, "result.json")
    if os.path.exists(result_path):
        print(f"[skip] {result_path} 已存在")
        return

    # 数据: 现场仿真 (与 prep 同 seed 语义一致); --data 可强制 (误吸收检验)
    s_base = kvlcc2()
    if args.data == "v1":
        s = s_base
    elif args.data == "ext":
        s = calibrate_ext(s_base, target_frac=args.cal_frac, seed=args.seed)
    elif args.variant == "v1":
        s = s_base
    else:
        s = calibrate_ext(s_base, target_frac=args.cal_frac, seed=args.seed)
    cases, var_names = build_cases(s, args.variant, MANEUVERS,
                                   noise_rel=args.noise, seed=args.seed, eq=args.eq)

    # LLM 客户端 + G_INIT 语义种子增强 (heuristic 后端的领域知识替身; 零内核改动).
    # 机制: 包装 llm_ops.G_INIT — 保留原调用, 输出后追加 MMG 结构种子 DSL.
    # 语义上等价于 LLM 的 domain prior (真实 LLM 后端同样会受益, 但其输出
    # 不可控, 故对 gponly 臂也注入程序化种子 — 与 asm 轮 wall_seeds 机制同型).
    bk = BACKENDS[args.backend]
    api_key = os.environ.get("LGF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    llm_client = make_client(
        bk["backend"],
        url=os.environ.get("LGF_LLM_URL", "https://api.openai.com/v1"),
        model=bk.get("llm_model"), api_key=api_key,
        temperature=0.7, timeout=60.0,
        max_tokens=bk.get("max_tokens"), extra_body=bk.get("extra_body"))

    from shipman.seeds import patch_ginit_with_mmg_seeds, mmg_seed_dsls
    patch_ginit_with_mmg_seeds(variant=args.variant, eq=args.eq)

    use_llm = (args.arm == "llm")
    # gponly 臂也走 G_INIT 通道 (heuristic 后端零 API 成本): 种子注入需要.
    # 消融口径: gponly = heuristic LLM + 程序化种子 (无语义 M_GUIDE);
    #          llm   = glm53 + M_GUIDE 全链路. 论文中如实标注.
    seed_run = zlib.crc32(f"sm|{args.variant}|{args.eq}|{args.noise}|{args.seed}|{args.arm}".encode()) % 2**31
    cfg = LGFConfig(
        population=args.pop, generations=args.gens,
        sub_sample=300, time_budget_s=args.budget,
        max_edges=16, max_params=9,
        use_llm=True, use_priors=True, use_attribution=use_llm,
        physics_context=PHYSICS_CTX_MMG if use_llm else "",
        global_params_only=True,
        seed=seed_run,
    )

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    out = run_lgf(cases, var_names, cfg, log_dir=run_dir, llm_client=llm_client,
                  name=f"{args.variant}_{args.eq}")
    result = {
        "best": out["best"], "top_final": out["top_final"][:3],
        "llm_stats": out.get("llm_stats"), "elapsed_s": time.time() - t0,
        "config": dict(variant=args.variant, eq=args.eq, noise=args.noise,
                       seed=args.seed, arm=args.arm, backend=args.backend,
                       budget=args.budget, cal_frac=args.cal_frac,
                       data_tag=args.data if args.data != "auto" else
                       ("v1" if args.variant == "v1" else "ext")),
        "data_meta": dict(ship="kvlcc2", maneuvers=list(MANEUVERS),
                          cDv=s.cDv, cDr=s.cDr, mem_A=list(s.mem_A)),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    print(f"[done] {result_path} loss={out['best']['loss']:.4f} "
          f"dsl={out['best']['dsl'][:80]}")


if __name__ == "__main__":
    main()
