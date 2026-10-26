"""P1b launcher: inject minimax key from EvoScientist config, then run
minimax x pope_wy (same flags as heuristic_pope_wy for a clean A/B).

v2 (2026-10-04):
- SetThreadExecutionState 阻止运行期间系统休眠 (v1 死因: 9/30 晚休眠杀进程)
- stdout/stderr 强制 UTF-8 (v1 runlog.err 中文 GBK 乱码)
Usage: python scripts/_run_minimax_wy.py   (background-friendly)
"""
import os
import re
import runpy
import sys

CFG = r"C:\Users\lenovo\.config\evoscientist\config.yaml"
REPO = r"d:\AI\LLM-Feynman-G"

# --- 防 CPU 在跑批期间进入休眠 (Windows only; 失败不影响主流程) ---
try:
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
except Exception as e:
    print(f"[launcher] SetThreadExecutionState failed: {e}", flush=True)

# --- 日志编码: 后台重定向时 PowerShell 默认解码 GBK 会乱码 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

key = None
with open(CFG, encoding="utf-8") as f:
    m = re.search(r"^minimax_api_key:\s*'([^']*)'\s*$|^minimax_api_key:\s*([^\s']\S*)\s*$",
                  f.read(), re.M)
if m:
    key = (m.group(1) or m.group(2) or "").strip().strip("'\"")
if not key:
    sys.exit("[launcher] minimax_api_key not found in evoscientist config")

os.environ["LGF_LLM_API_KEY"] = key
os.environ["LGF_LLM_DEBUG"] = "1"  # backend failures -> stderr (visible in runlog.err)

sys.argv = [
    "rc_asm_run.py",
    "--scheme", "B",
    "--backend", "minimax",
    "--tag", "minimax_pope_wy",
    "--dataset", "dataset_pope_wy.npz",
    "--fitness", "stress",
    "--fit_mask", "valid_fit",
    "--wall_boost", "8",
    "--w_vw", "5",
    "--w_uw", "3",
    "--budget", "150",
]
runpy.run_path(os.path.join(REPO, "scripts", "rc_asm_run.py"), run_name="__main__")
