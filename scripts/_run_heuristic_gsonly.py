"""launcher: heuristic x pope_gsonly (GS 换基单独, 归因实验), 防休眠 + UTF-8."""
import ctypes
import os
import runpy
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.argv = [
    "rc_asm_run.py",
    "--scheme", "B", "--backend", "heuristic",
    "--tag", "heuristic_gsonly",
    "--dataset", "dataset_pope_gsonly.npz",
    "--fitness", "stress",
    "--fit_mask", "valid_fit",
    "--wall_boost", "8", "--w_vw", "5", "--w_uw", "3",
    "--budget", "150",
]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "rc_asm_run.py"), run_name="__main__")
