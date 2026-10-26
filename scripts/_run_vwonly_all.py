# -*- coding: utf-8 -*-
"""_run_vwonly_all.py — vw-only 跑批驱动器.

3 条并行流水线, 每条内部串行:
  pysr: PySR   x {v,g,e,m}
  go  : lgf_go x {v,g,e,m}  (heuristic, ginit_only)
  mg  : lgf_mg x {v,g,e,m}  (heuristic, full M_GUIDE)

用法:
  python -X utf8 scripts/_run_vwonly_all.py --stream pysr
  python -X utf8 scripts/_run_vwonly_all.py --stream go
  python -X utf8 scripts/_run_vwonly_all.py --stream mg
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
VARIANTS = ["v", "g", "e", "m"]


def run_one(variant: str, method: str, log: Path) -> bool:
    cmd = [PY, "-X", "utf8", str(REPO / "scripts" / "rc_asm_vw_only.py"),
           "--variant", variant, "--method", method,
           "--backend", "heuristic"]
    t0 = time.time()
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n===== {method}/{variant} start {time.strftime('%H:%M:%S')} =====\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    ok = (r.returncode == 0)
    print(f"[{method}/{variant}] rc={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["pysr", "go", "mg"], required=True)
    args = ap.parse_args()

    method = {"pysr": "pysr", "go": "lgf_go", "mg": "lgf_mg"}[args.stream]
    log = REPO / "outputs" / "asm" / f"_vwonly_stream_{args.stream}.log"

    fails = []
    for v in VARIANTS:
        try:
            if not run_one(v, method, log):
                fails.append(f"{method}/{v}")
        except Exception as e:
            print(f"[{method}/{v}] EXC {type(e).__name__}: {e}", flush=True)
            fails.append(f"{method}/{v}")

    print(f"[stream {args.stream}] DONE fails={fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
