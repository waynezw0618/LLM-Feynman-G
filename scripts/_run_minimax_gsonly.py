"""P1c launcher: inject minimax key from EvoScientist config, then run
minimax x pope_gsonly (gsonly=GS-only arm, 4-arm 验证中最优基线).

与 _run_minimax_wy.py v2 (P1b) 同框架, 只换 dataset/tag:
  - dataset: dataset_pope_gsonly.npz  (9 列: 8 v-不变量 + yw, T5g 基底)
  - tag:     minimax_pope_gsonly

种子由 rc_asm_run.py:152 用 crc32("B|T1") 等 hash target 名决定,
与 heuristic_pope_gsonly 完全一致 -> 单因子公平对照.

v1 (2026-10-08): P1c 启跑
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
    "--tag", "minimax_pope_gsonly",
    "--dataset", "dataset_pope_gsonly.npz",
    "--fitness", "stress",
    "--fit_mask", "valid_fit",
    "--wall_boost", "8",
    "--w_vw", "5",
    "--w_uw", "3",
    "--budget", "150",
]
runpy.run_path(os.path.join(REPO, "scripts", "rc_asm_run.py"), run_name="__main__")
