# -*- coding: utf-8 -*-
"""_run_v4_matrix.py — V4 轮 12 组矩阵跑批 launcher.

矩阵: {v,g,e,m}(vw-only, yw 标准化 [0,1]) x {lgf_go, lgf_mg, pysr}
  - lgf_go / lgf_mg: backend=glm53 (GLM 5.3, thinking low, 16k max_tokens)
  - pysr: 同特征池同尺度 (公平对照)

V4 修复 (2026-10-12, 本轮全部生效):
  L1-1 DSL/参数对齐: walk preorder + _normalize_params 保名 + params() 按名排序
  L1-2 paramfit 硬限时 (fit_joint time_budget_s + deadline callback)
  L1-3 解析率: glm-5.3 thinking enabled + reasoning_effort=low (m 变体
      G_INIT 解析率 6/30 -> 30/30)
  L1-4 yw 标准化: yw=(1-|yh|) in [0,1] (旧 [0,180] 把 theta 压到 1e-7)
  L1-5 paramfit 多 start: 对数尺度 start 策略 (paramfit V4)

输出: outputs/asm/runs_vwonly_v4_{lgf_go,lgf_mg,pysr}/{v,g,e,m}/result.json
用法: python -X utf8 scripts/_run_v4_matrix.py [--only lgf_go|lgf_mg|pysr|all]
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIPU_KEY = "ad3c10eadc414601b6a9dd815d5aaf9e.YuNDwUYg9fLLsJCB"

VARIANTS = ["v", "g", "e", "m"]
METHODS = ["lgf_go", "lgf_mg", "pysr"]
BUDGET = 600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all"] + METHODS + [f"{m}:{v}" for m in METHODS
                                                 for v in VARIANTS])
    args = ap.parse_args()

    env = dict(os.environ)
    env["LGF_LLM_API_KEY"] = ZHIPU_KEY
    env["PYTHONIOENCODING"] = "utf-8"

    jobs = []
    for m in METHODS:
        for v in VARIANTS:
            key = f"{m}:{v}"
            if args.only in ("all", m, key):
                jobs.append((m, v))

    print(f"[v4-matrix] {len(jobs)} jobs: {jobs}", flush=True)
    t_all = time.time()
    for i, (m, v) in enumerate(jobs):
        out_dir = os.path.join(REPO, "outputs", "asm", f"runs_vwonly_v4_{m}", v)
        marker = os.path.join(out_dir, "result.json")
        if os.path.exists(marker):
            print(f"[v4-matrix] ({i+1}/{len(jobs)}) SKIP {m}:{v} "
                  f"(result.json exists)", flush=True)
            continue
        cmd = [sys.executable, "-X", "utf8",
               os.path.join(REPO, "scripts", "rc_asm_vw_only.py"),
               "--variant", v, "--method", m, "--add_yw", "--tag", "_v4"]
        if m != "pysr":
            cmd += ["--backend", "glm53", "--budget", str(BUDGET)]
        print(f"[v4-matrix] ({i+1}/{len(jobs)}) RUN {m}:{v} :: "
              f"{' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        with open(os.path.join(REPO, f"_v4_{m}_{v}.log"), "w",
                  encoding="utf-8") as lf:
            r = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf,
                               stderr=subprocess.STDOUT)
        print(f"[v4-matrix] ({i+1}/{len(jobs)}) DONE {m}:{v} "
              f"rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
    print(f"[v4-matrix] ALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
