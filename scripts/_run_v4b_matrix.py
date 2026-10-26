# -*- coding: utf-8 -*-
"""_run_v4b_matrix.py — V4b 轮矩阵跑批 (GP 缓存 bug 修复 + log-law 提示后重跑).

背景 (2026-10-12):
  V4a (tag _v4) 的 lgf_* 结果 0.25-0.30, 比上一轮 yw 轮 LGF 0.085 明显退化.
  诊断出两个根因:
    (1) Graph.clone() 拷贝父代 loss/theta, 而 GP crossover/mutate 全用 clone
        构造子代 -> V4 存活者缓存误判子代为存活者跳过拟合 -> GP 进化失效
        (每代 0.7s, 40 代 0.408->0.297, top_final 坍缩到 1-2 家族).
    (2) G_INIT prompt (v3a) 只提示 tanh/exp 阻尼, 把 log(yw) 对数律类结构
        挤出候选池; 而上一轮 LGF(0.085) 与 PySR(0.033) 的获胜公式都用 log(yw).
  V4b 修复: pipeline_lgf 子代 loss/theta 重置 + 预算预留 25% 给 final refit;
            llm_ops G_INIT_wall_hint/fewshot/硬规则补 log(yw) 对数律家族.

矩阵 (2 路并行):
  流 A: {lgf_go, lgf_mg} x {v,g,e,m}, tag=_v4b, backend=glm53, budget=600
  流 B: pysr x {v,g,e,m}, tag=_v4 (原样, PySR 路径不受 LGF 修复影响)
输出:
  outputs/asm/runs_vwonly_v4b_{lgf_go,lgf_mg}/{v,g,e,m}/
  outputs/asm/runs_vwonly_v4_pysr/{v,g,e,m}/
用法: python -X utf8 scripts/_run_v4b_matrix.py
"""
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZHIPU_KEY = "ad3c10eadc414601b6a9dd815d5aaf9e.YuNDwUYg9fLLsJCB"

VARIANTS = ["v", "g", "e", "m"]
BUDGET = 600.0

# (method, variant, tag)
STREAM_A = [(m, v, "_v4b") for m in ["lgf_go", "lgf_mg"] for v in VARIANTS]
STREAM_B = [("pysr", v, "_v4") for v in VARIANTS]


def _job_cmd(method, variant, tag):
    cmd = [sys.executable, "-X", "utf8",
           os.path.join(REPO, "scripts", "rc_asm_vw_only.py"),
           "--variant", variant, "--method", method, "--add_yw", "--tag", tag]
    if method != "pysr":
        cmd += ["--backend", "glm53", "--budget", str(BUDGET)]
    return cmd


def _out_dir(method, variant, tag):
    return os.path.join(REPO, "outputs", "asm", f"runs_vwonly{tag}_{method}", variant)


def _run_stream(name, jobs, env):
    log_lines = []
    for i, (method, variant, tag) in enumerate(jobs):
        out_dir = _out_dir(method, variant, tag)
        marker = os.path.join(out_dir, "result.json")
        if os.path.exists(marker):
            msg = f"[{name}] ({i+1}/{len(jobs)}) SKIP {method}:{variant}"
            print(msg, flush=True)
            log_lines.append(msg)
            continue
        cmd = _job_cmd(method, variant, tag)
        msg = f"[{name}] ({i+1}/{len(jobs)}) RUN {method}:{variant}"
        print(msg, flush=True)
        log_lines.append(msg)
        t0 = time.time()
        with open(os.path.join(REPO, f"_v4b_{name}_{method}_{variant}.log"), "w",
                  encoding="utf-8") as lf:
            r = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf,
                               stderr=subprocess.STDOUT)
        done = (f"[{name}] ({i+1}/{len(jobs)}) DONE {method}:{variant} "
                f"rc={r.returncode} {time.time()-t0:.0f}s")
        print(done, flush=True)
        log_lines.append(done)
    return log_lines


def main():
    env = dict(os.environ)
    env["LGF_LLM_API_KEY"] = ZHIPU_KEY
    env["PYTHONIOENCODING"] = "utf-8"
    # 避免 BLAS 线程过度订阅 (2 路并行)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"

    print(f"[v4b] stream A (LGF) = {STREAM_A}", flush=True)
    print(f"[v4b] stream B (PySR) = {STREAM_B}", flush=True)
    t_all = time.time()

    pa = subprocess.Popen([sys.executable, "-X", "utf8", __file__, "--worker", "A"],
                          cwd=REPO, env=env)
    pb = subprocess.Popen([sys.executable, "-X", "utf8", __file__, "--worker", "B"],
                          cwd=REPO, env=env)
    pa.wait()
    pb.wait()
    print(f"[v4b] ALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


def worker(which):
    env = dict(os.environ)
    env["LGF_LLM_API_KEY"] = ZHIPU_KEY
    env["PYTHONIOENCODING"] = "utf-8"
    jobs = STREAM_A if which == "A" else STREAM_B
    _run_stream(which, jobs, env)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
