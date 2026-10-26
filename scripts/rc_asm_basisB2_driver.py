# -*- coding: utf-8 -*-
"""rc_asm_basisB2_driver.py — 方案B v2 (张量基系数) 全矩阵串行驱动.

任务矩阵: variant ∈ {v,g,e,m} × method ∈ {pysr, lgf_mg, lgf_go} = 12 组.
每组对**每一条张量基系数 a_m(Θ)** 做符号回归, 再重组 b23 并评估 (同一 NMSE).

用法:
  python -X utf8 scripts/rc_asm_basisB2_driver.py --mode pysr
  python -X utf8 scripts/rc_asm_basisB2_driver.py --mode lgf --budget 100
  python -X utf8 scripts/rc_asm_basisB2_driver.py --mode lgf --wait_log <log>
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
VARIANTS = ("v", "g", "e", "m")


def read_key(name):
    for line in CFG.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"^{name}:\s*(\S+)\s*$", line)
        if m:
            return m.group(1)
    raise SystemExit(f"{name} not found in {CFG}")


def wait_for(path: Path, token="ALL DONE", timeout=14400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists() and token in path.read_text(encoding="utf-8",
                                                     errors="ignore"):
            print(f"[basisB2-driver] gate open", flush=True)
            return True
        time.sleep(15)
    print(f"[basisB2-driver] gate timeout, 继续启动", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pysr", "lgf"], required=True)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--wait_log", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pysr_maxsize", type=int, default=25)
    ap.add_argument("--prompt", default="physics")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=35)
    ap.add_argument("--maxparams", type=int, default=8)
    a = ap.parse_args()

    env = dict(os.environ)
    if a.mode == "lgf":
        key = read_key("zhipu_api_key")
        env["LGF_LLM_API_KEY"] = key
        print(f"[basisB2-driver] key set (tail ...{key[-4:]})", flush=True)
        if a.wait_log:
            wait_for(REPO / a.wait_log)

    variants = ([v.strip() for v in a.variants.split(",") if v.strip()]
                or list(VARIANTS))
    if a.mode == "pysr":
        methods = ["pysr"]
    elif a.methods:
        methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    else:
        methods = ["lgf_mg", "lgf_go"]
    jobs = [(v, m) for m in methods for v in variants]
    t00 = time.time()
    for k, (variant, method) in enumerate(jobs, 1):
        out = (REPO / "outputs" / "asm" / "runs_basisB2" /
               f"{variant}_{method}{a.tag}" / "result.json")
        if out.exists():
            print(f"[basisB2-driver {k}/{len(jobs)}] SKIP {variant}/{method}{a.tag}",
                  flush=True)
            continue
        print(f"[basisB2-driver {k}/{len(jobs)}] RUN {variant}/{method}{a.tag} @ "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        cmd = [PY, "-X", "utf8", "-W", "ignore",
               str(REPO / "scripts" / "rc_asm_basisB2.py"),
               "--stage", "sr", "--variant", variant, "--method", method,
               "--cnorm", "percase", "--whiten", "0",
               "--budget", str(a.budget),
               "--pysr_maxsize", str(a.pysr_maxsize),
               "--prompt", a.prompt,
               "--pop", str(a.pop), "--gens", str(a.gens),
               "--maxparams", str(a.maxparams)]
        if a.tag:
            cmd += ["--tag", a.tag]
        if method != "pysr":
            cmd += ["--backend", "glm53"]
        r = subprocess.run(cmd, cwd=str(REPO), env=env)
        print(f"[basisB2-driver {k}/{len(jobs)}] DONE {variant}/{method}{a.tag} "
              f"rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
    print(f"[basisB2-driver] ALL DONE total={time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
