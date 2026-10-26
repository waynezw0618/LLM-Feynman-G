# -*- coding: utf-8 -*-
"""rc_asm_vw_t23_driver.py — T23 全量轮串行驱动 (LGF 8 组 或 PySR 4 组).

用法:
  python -X utf8 scripts/rc_asm_vw_t23_driver.py --mode lgf
  python -X utf8 scripts/rc_asm_vw_t23_driver.py --mode pysr
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


def read_key(name: str) -> str:
    for line in CFG.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"^{name}:\s*(\S+)\s*$", line)
        if m:
            return m.group(1)
    raise SystemExit(f"{name} not found in {CFG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lgf", "pysr"], required=True)
    ap.add_argument("--tag", default="", help="输出目录后缀 (如 _ctl)")
    args = ap.parse_args()

    key = read_key("zhipu_api_key")
    os.environ["LGF_LLM_API_KEY"] = key
    print(f"[driver] key set (tail ...{key[-4:]}) tag='{args.tag}'", flush=True)

    if args.mode == "lgf":
        jobs = [(v, m) for m in ("lgf_mg", "lgf_go") for v in ("v", "g", "e", "m")]
    else:
        jobs = [(v, "pysr") for v in ("v", "g", "e", "m")]

    t00 = time.time()
    for k, (variant, method) in enumerate(jobs, 1):
        out_json = (REPO / "outputs" / "asm" / f"runs_vw_t23full_{method}{args.tag}"
                    / variant / "result.json")
        if out_json.exists():
            print(f"[driver {k}/{len(jobs)}] SKIP {method}/{variant} (已有 result.json)",
                  flush=True)
            continue
        print(f"[driver {k}/{len(jobs)}] RUN {method}/{variant} @ {time.strftime('%H:%M:%S')}",
              flush=True)
        t0 = time.time()
        cmd = [PY, "-X", "utf8", str(REPO / "scripts" / "rc_asm_vw_t23_full.py"),
               "--variant", variant, "--method", method, "--backend", "glm53"]
        if args.tag:
            cmd += ["--tag", args.tag]
        r = subprocess.run(cmd, cwd=str(REPO))
        print(f"[driver {k}/{len(jobs)}] DONE {method}/{variant} "
              f"rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
    print(f"[driver] ALL DONE total={time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
