# -*- coding: utf-8 -*-
"""rc_asm_basisB_driver.py — 方案B (纯张量基系数回归) LGF 串行驱动.

任务矩阵: variant ∈ {g,e} × target ∈ {a1,a5,a6,a10} × method ∈ {lgf_mg,lgf_go}
= 16 组. 每组对一条系数 a_m(Θ) 做符号回归, 结束后写入
outputs/asm/runs_basisB/{variant}_{method}/result.json

--wait_log 可指定一个日志文件, 等其中出现 'ALL DONE' 再启动 (避免与其它跑批争 API).

用法:
  python -X utf8 scripts/rc_asm_basisB_driver.py --mode lgf
  python -X utf8 scripts/rc_asm_basisB_driver.py --mode lgf --wait_log outputs/asm/_t23full_ctl_driver.log
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
CFG = Path(r"C:\Users\lenovo\.config\evoscientist\config.yaml")


def read_key(name):
    for line in CFG.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"^{name}:\s*(\S+)\s*$", line)
        if m:
            return m.group(1)
    raise SystemExit(f"{name} not found in {CFG}")


def wait_for(path: Path, token="ALL DONE", timeout=14400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists() and token in path.read_text(encoding="utf-8", errors="ignore"):
            print(f"[basisB-driver] gate open ({path} 含 '{token}')", flush=True)
            return True
        time.sleep(20)
    print(f"[basisB-driver] gate timeout, 继续启动", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lgf"], default="lgf")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--wait_log", default="")
    a = ap.parse_args()

    key = read_key("zhipu_api_key")
    os.environ["LGF_LLM_API_KEY"] = key
    print(f"[basisB-driver] key set (tail ...{key[-4:]})", flush=True)

    if a.wait_log:
        wait_for(REPO / a.wait_log)

    jobs = [(v, m) for m in ("lgf_mg", "lgf_go") for v in ("g", "e")]
    t00 = time.time()
    for k, (variant, method) in enumerate(jobs, 1):
        out = REPO / "outputs" / "asm" / "runs_basisB" / f"{variant}_{method}" / "result.json"
        if out.exists():
            print(f"[basisB-driver {k}/{len(jobs)}] SKIP {variant}/{method}", flush=True)
            continue
        print(f"[basisB-driver {k}/{len(jobs)}] RUN {variant}/{method} @ "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [PY, "-X", "utf8", str(REPO / "scripts" / "rc_asm_vw_t23_basisB.py"),
             "--stage", "sr", "--variant", variant, "--method", method,
             "--backend", "glm53", "--budget", str(a.budget)],
            cwd=str(REPO))
        print(f"[basisB-driver {k}/{len(jobs)}] DONE {variant}/{method} rc={r.returncode} "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[basisB-driver] ALL DONE total={time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
