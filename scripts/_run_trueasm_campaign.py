# -*- coding: utf-8 -*-
"""_run_trueasm_campaign.py — 真 ASM 全量跑批 launcher (PySR + LGF/mm).

跑批矩阵 (2026-10-09):
  阶段1 PySR 纯符号回归: {v,g,e} × 1 (niterations=50, maxsize=30)
  阶段2 LGF + MiniMax-M2.7: {v,g,e} × {mg, go} (budget=800s)
顺序执行避免 CPU 争抢; 已有 result.json 的 run 自动跳过 (断点续跑).

用法 (分离启动):
  Start-Process python -ArgumentList "-X","utf8",
      "d:\\AI\\LLM-Feynman-G\\scripts\\_run_trueasm_campaign.py"
进度: outputs/asm/_trueasm_progress.log
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time

REPO = r"d:\AI\LLM-Feynman-G"
CFG = r"C:\Users\lenovo\.config\evoscientist\config.yaml"
PY = sys.executable
OUT = os.path.join(REPO, "outputs", "asm")

# --- 防休眠 (Windows only; 失败不影响主流程) ---
try:
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
except Exception as e:
    print(f"[launcher] SetThreadExecutionState failed: {e}", flush=True)

# --- MiniMax key (与 v3 launcher 同源) ---
key = None
with open(CFG, encoding="utf-8") as f:
    m = re.search(r"^minimax_api_key:\s*'([^']*)'\s*$|^minimax_api_key:\s*([^\s']\S*)\s*$",
                  f.read(), re.M)
if m:
    key = (m.group(1) or m.group(2) or "").strip().strip("'\"")
if not key:
    sys.exit("[launcher] minimax_api_key not found in evoscientist config")

PSYSR_RUNS = [("v",), ("g",), ("e",)]
LGF_RUNS = [(v, arm) for v in ("v", "g", "e") for arm in ("mg", "go")]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "_trueasm_progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, log_name, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    lf = os.path.join(OUT, f"_trueasm_{log_name}.log")
    t0 = time.time()
    with open(lf, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           cwd=REPO, env=env)
    return p.returncode, (time.time() - t0) / 60


def main():
    t_all = time.time()
    log("=== true-ASM 跑批启动: PySR{v,g,e} + LGF/mm{v,g,e}x{mg,go} ===")

    # ---- 阶段1: PySR ----
    for (v,) in PSYSR_RUNS:
        out_json = os.path.join(OUT, "runs_trueasm_pysr", v, "result.json")
        if os.path.exists(out_json):
            log(f"[skip] pysr/{v}: result.json 已存在")
            continue
        cmd = [PY, "-X", "utf8", os.path.join(REPO, "scripts", "rc_asm_true_pysr.py"),
               "--variant", v, "--niterations", "50", "--maxsize", "30",
               "--populations", "8", "--population_size", "40", "--seed", "0"]
        rc, dt = run(cmd, f"pysr_{v}")
        log(f"[{'done' if rc == 0 else 'FAIL'}] pysr/{v}: exit={rc} {dt:.1f} min")

    # ---- 阶段2: LGF + MiniMax M2.7 ----
    for v, arm in LGF_RUNS:
        out_json = os.path.join(OUT, "runs_trueasm_lgf", f"{v}_{arm}", "result.json")
        if os.path.exists(out_json):
            log(f"[skip] lgf/{v}_{arm}: result.json 已存在")
            continue
        cmd = [PY, "-X", "utf8", os.path.join(REPO, "scripts", "rc_asm_true_lgf.py"),
               "--variant", v, "--arm", arm, "--backend", "minimax",
               "--budget", "800", "--pop", "60", "--gens", "40",
               "--sub_sample", "300", "--max_edges", "16", "--max_params", "6"]
        rc, dt = run(cmd, f"lgf_{v}_{arm}",
                     env_extra={"LGF_LLM_API_KEY": key, "LGF_LLM_DEBUG": "1"})
        log(f"[{'done' if rc == 0 else 'FAIL'}] lgf/{v}_{arm}: exit={rc} {dt:.1f} min")

    log(f"=== 全部完成, 总耗时 {(time.time() - t_all) / 3600:.2f} h ===")


if __name__ == "__main__":
    main()
