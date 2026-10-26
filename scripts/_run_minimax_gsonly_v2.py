# -*- coding: utf-8 -*-
"""P1d launcher: 改进后的 MiniMax-M2.7 x pope_gsonly (P0->P2 全套改进落地).

改进清单 (相对 P1c):
  A1 attribution per-case 误差分解 + 定性标签 (EXPLODED/SIGN-FLIP/区域)
  A2 M_GUIDE 硬规则: 目标值域/符号 + 危险分母 (vanishing 特征清单)
  A3 M_GUIDE 硬规则: 最差工况行 (报告级, 不改 fitness — 用户明确不改目标函数)
  B1 精英保护: M_GUIDE 变异体追加参赛, 不顶替精英 (pipeline_lgf.py)
  B2 G_INIT yw 约束 + 程序化 wall-seed 注入 (冷启动保证近壁通道)
  B3 T5g 禁用 sub/div (纯幂律先验, apply_mutations 节点级硬拦截)
  (C2 ginit_only 本臂不启用 — 那是对照臂, 留给后续)

对照: heuristic_v2 同管线同参数同 forbid_map 同 seed (单因子=后端),
     以及 P1c 两臂 (旧管线) 做跨管线归因.
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
os.environ["LGF_LLM_DEBUG"] = "1"

sys.argv = [
    "rc_asm_run.py",
    "--scheme", "B",
    "--backend", "minimax",
    "--tag", "minimax_pope_gsonly_v2",
    "--dataset", "dataset_pope_gsonly.npz",
    "--fitness", "stress",
    "--fit_mask", "valid_fit",
    "--wall_boost", "8",
    "--w_vw", "5",
    "--w_uw", "3",
    "--budget", "150",
    "--forbid_map", "T5g:sub,div",
]
runpy.run_path(os.path.join(REPO, "scripts", "rc_asm_run.py"), run_name="__main__")
