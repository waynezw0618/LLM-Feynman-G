"""launcher: 补跑 heuristic x pope_gsrt 剩余目标 T4v/T5g + 合并 5 目标 summary.

背景: 2026-10-06 18:4x 启动的 gsrt 全量跑在 T3v 完成后进程死亡 (shell 会话
崩溃连带), runs_heuristic_pope_gsrt/ 下已有 T1/T2v/T3v 的 final.json.
本脚本:
  1. runpy 调 rc_asm_run.py --targets T4v T5g (同 tag, 同 dir, 不覆盖已完成)
     —— 注意它会写一个只含 2 目标的 summary_B.json, 随后被步骤 2 覆盖
  2. 从 5 个 B_a_{tgt}_final.json 重建完整 summary_B.json (含 n_g/n_c)
防休眠 + UTF-8, 与 _run_minimax_wy.py v2 同套路.
"""
import ctypes
import json
import os
import runpy
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUN_DIR = os.path.join(REPO, "outputs", "asm", "runs_heuristic_pope_gsrt")
SEL = ["T1", "T2v", "T3v", "T4v", "T5g"]

sys.argv = [
    "rc_asm_run.py",
    "--scheme", "B", "--backend", "heuristic",
    "--tag", "heuristic_pope_gsrt",
    "--dataset", "dataset_pope_gsrt.npz",
    "--fitness", "stress",
    "--fit_mask", "valid_fit",
    "--wall_boost", "8", "--w_vw", "5", "--w_uw", "3",
    "--budget", "150",
    "--targets", "T4v", "T5g",
]
runpy.run_path(os.path.join(HERE, "rc_asm_run.py"), run_name="__main__")

# ---- 合并 5 目标 summary ----
sys.path.insert(0, REPO)
from lgf import dsl as lgfdsl  # noqa: E402
import numpy as np  # noqa: E402

ds = np.load(os.path.join(REPO, "outputs", "asm", "dataset_pope_gsrt.npz"))
var_names = [str(x) for x in ds["featB_keys"]]

summary = dict(scheme="B", backend="heuristic", tag="heuristic_pope_gsrt",
               model=None, var_names=var_names, budget=150.0, n_cases=5,
               fitness="stress", dataset="dataset_pope_gsrt.npz",
               fit_mask="valid_fit", wall_boost=8.0, wall_region=0.7,
               comp_w={c: w for c, w in zip(
                   ["uu", "vv", "ww", "uv", "uw", "vw"],
                   [1.0, 1.0, 1.0, 1.0, 3.0, 5.0])},
               targets={})
for tgt in SEL:
    fp = os.path.join(RUN_DIR, f"B_a_{tgt}_final.json")
    if not os.path.exists(fp):
        print(f"[merge][WARN] 缺 {fp}, 该目标将缺席 summary")
        continue
    best = json.load(open(fp, encoding="utf-8"))["best"]
    n_g = n_c = 0
    try:
        g = lgfdsl.parse_dsl(best["dsl"], var_names)
        n_g = len(g.global_names()); n_c = len(g.case_names())
    except Exception:
        pass
    summary["targets"][tgt] = dict(
        dsl=best["dsl"], loss=best["loss"], complexity=best["complexity"],
        theta=best["theta"], metrics=best["metrics"], n_g=n_g, n_c=n_c,
        elapsed_s=None)
    print(f"[merge] {tgt}: loss={best['loss']:.6g} :: {best['dsl'][:100]}")

out = os.path.join(RUN_DIR, "summary_B.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"[merge] -> {out} ({len(summary['targets'])}/5 目标)")
print("[done]", time.strftime("%H:%M:%S"))
