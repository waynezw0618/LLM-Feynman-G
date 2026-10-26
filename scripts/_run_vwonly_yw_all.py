# -*- coding: utf-8 -*-
"""_run_vwonly_yw_all.py — vw-only + yw 特征全量跑批驱动.

矩阵: {v,g,e,m} × {pysr, lgf_go, lgf_mg} = 12 组合
  - 特征池: 原 17/43 维 + yw = (1-|yh|)*180 (真壁距)
  - LGF 后端: minimax (MiniMax-M2.7 真实 API, 不用 heuristic)
  - key 来源: C:/Users/lenovo/.config/evoscientist/config.yaml (minimax_api_key 行)
  - 每组硬超时 (subprocess timeout), 超时杀进程并记录 TIMEOUT 后继续
  - skip-if-exists: result.json 已存在则跳过 (断点续跑)

输出:
  outputs/asm/runs_vwonly_yw_{method}/{variant}/result.json
  outputs/asm/_vwonly_yw_{method}_{variant}.log
  outputs/asm/_vwonly_yw_summary.json
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"d:\AI\LLM-Feynman-G")
PY = r"C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe"
CFG_YAML = Path(r"C:\Users\lenovo\.config\evoscientist\config.yaml")
OUT = REPO / "outputs" / "asm"

VARIANTS = ["v", "g", "e", "m"]
METHODS = ["pysr", "lgf_go", "lgf_mg"]

# 硬超时 (秒): pysr 45 min; lgf v/g/e 40 min; lgf m 60 min (43+1 维最慢)
TIMEOUT = {
    "pysr": 2700,
    "lgf_go": dict(v=2400, g=2400, e=2400, m=3600),
    "lgf_mg": dict(v=2400, g=2400, e=2400, m=3600),
}


def read_key() -> str:
    for line in CFG_YAML.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("minimax_api_key:"):
            key = line.split(":", 1)[1].strip().strip('"').strip("'")
            if key:
                return key
    raise SystemExit("minimax_api_key not found in config.yaml")


def main():
    key = read_key()
    env = dict(os.environ)
    env["LGF_LLM_API_KEY"] = key
    env["PYTHONIOENCODING"] = "utf-8"
    env["LGF_LLM_DEBUG"] = "1"

    summary = {}
    t_all = time.time()
    for method in METHODS:
        for variant in VARIANTS:
            tag = f"{method}/{variant}"
            run_dir = OUT / f"runs_vwonly_yw_{method}" / variant
            result_file = run_dir / "result.json"
            if result_file.exists():
                print(f"[skip] {tag}: result.json 已存在", flush=True)
                summary[tag] = dict(status="skipped")
                continue

            cmd = [PY, "-X", "utf8", "scripts/rc_asm_vw_only.py",
                   "--variant", variant, "--method", method,
                   "--add_yw", "--tag", "_yw"]
            if method != "pysr":
                cmd += ["--backend", "minimax"]

            to = TIMEOUT[method] if method == "pysr" else TIMEOUT[method][variant]
            log_file = OUT / f"_vwonly_yw_{method}_{variant}.log"
            print(f"[run] {tag}: timeout={to}s, log={log_file.name}", flush=True)
            t0 = time.time()
            status = "ok"
            with open(log_file, "w", encoding="utf-8") as lf:
                try:
                    r = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf,
                                       stderr=subprocess.STDOUT, timeout=to)
                    status = "ok" if r.returncode == 0 else f"exit={r.returncode}"
                except subprocess.TimeoutExpired:
                    status = "TIMEOUT"
            elapsed = round(time.time() - t0, 1)
            summary[tag] = dict(status=status, elapsed_s=elapsed,
                                done=result_file.exists())
            print(f"[done] {tag}: {status} in {elapsed}s "
                  f"(result.json={'yes' if result_file.exists() else 'NO'})",
                  flush=True)
            (OUT / "_vwonly_yw_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8")

    print(f"\n[ALL] total {round((time.time()-t_all)/60,1)} min", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v['status']} ({v.get('elapsed_s', 0)}s)")


if __name__ == "__main__":
    main()
