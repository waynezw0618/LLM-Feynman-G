# -*- coding: utf-8 -*-
"""v3 消融 launcher: 2x2x2 = {heur, MiniMax-M2.7} x {full, bare} x {M_GUIDE on, off}.

回答核心问题: "不计共享经验, heuristic 是否不敌 M2.7" (裸对裸、通道全开).

臂定义 (8 臂 x 5 目标 T1/T2v/T3v/T4v/T5g, 单因子差异 = 后端/机制/通道):
  full = stress fitness + valid_fit + wall_boost 8 + w_vw 5 + w_uw 3
         + forbid_map T5g:sub,div + B2 wall-seeds       (v2/P1c 全套共享机制)
  bare = coef fitness + valid mask + 全默认权重 + 无 forbid + 无 wall-seeds
         (剥离两臂共享的全部"经验"机制)
  mg   = M_GUIDE 每 5 代中途干预开;  go = --ginit_only (LLM 仅出初始种群)

两臂同享保留 (管线正确性修复, 非经验): V3-0 clone theta / V3-a llm_stats /
V3-b eval 层 forbidden / V3-c 值域护栏 / V3-d apply_mutations None /
G_INIT prompt 自造函数名硬规则 (prompt 属后端固有属性).

设计要点:
- 每臂独立子进程 (fresh import, 满足"老进程全杀重启"语义; 单臂崩溃不毁全局)
- 断点续跑: 该臂 summary_B.json 已存在则跳过
- seed 由 crc32(scheme|target) 决定, 8 臂同目标同 seed → 配对可比
- 臂序: bare 对决优先 (用户问题的主判据最先出来)

用法: python -X utf8 scripts/_run_v3_ablation.py
日志: outputs/asm/_v3_ablation_progress.log + 每臂 _v3_ablation_<tag>.log
"""
import os
import re
import subprocess
import sys
import time

REPO = r"d:\AI\LLM-Feynman-G"
CFG = r"C:\Users\lenovo\.config\evoscientist\config.yaml"
PY = sys.executable

# --- 防休眠 (Windows only; 失败不影响主流程) ---
try:
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
except Exception as e:
    print(f"[launcher] SetThreadExecutionState failed: {e}", flush=True)

BARE = ["--fitness", "coef", "--fit_mask", "valid", "--wall_seeds", "0"]
FULL = ["--fitness", "stress", "--fit_mask", "valid_fit",
        "--wall_boost", "8", "--w_vw", "5", "--w_uw", "3",
        "--forbid_map", "T5g:sub,div", "--wall_seeds", "1"]
GO = ["--ginit_only"]

# 臂序: bare 对决优先 (主判据), 然后 full 对决, 各自先 mg 后 go
ARMS = [
    ("v3_heur_bare_mg", "heuristic", BARE),
    ("v3_mm_bare_mg",   "minimax",   BARE),
    ("v3_heur_bare_go", "heuristic", BARE + GO),
    ("v3_mm_bare_go",   "minimax",   BARE + GO),
    ("v3_heur_full_mg", "heuristic", FULL),
    ("v3_mm_full_mg",   "minimax",   FULL),
    ("v3_heur_full_go", "heuristic", FULL + GO),
    ("v3_mm_full_go",   "minimax",   FULL + GO),
]

key = None
with open(CFG, encoding="utf-8") as f:
    m = re.search(r"^minimax_api_key:\s*'([^']*)'\s*$|^minimax_api_key:\s*([^\s']\S*)\s*$",
                  f.read(), re.M)
if m:
    key = (m.group(1) or m.group(2) or "").strip().strip("'\"")
if not key:
    sys.exit("[launcher] minimax_api_key not found in evoscientist config")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(REPO, "outputs", "asm", "_v3_ablation_progress.log"),
              "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    t_all = time.time()
    log(f"=== v3 消融跑批启动: 8 臂 x 5 目标, 顺序 {[a[0] for a in ARMS]} ===")
    for tag, backend, extra in ARMS:
        out_dir = os.path.join(REPO, "outputs", "asm", f"runs_{tag}")
        done = os.path.join(out_dir, "summary_B.json")
        if os.path.exists(done):
            log(f"[skip] {tag}: summary_B.json 已存在, 断点续跑跳过")
            continue
        cmd = [PY, "-X", "utf8", os.path.join(REPO, "scripts", "rc_asm_run.py"),
               "--scheme", "B", "--backend", backend, "--tag", tag,
               "--dataset", "dataset_pope_gsonly.npz", "--budget", "150"] + extra
        env = dict(os.environ)
        if backend == "minimax":
            env["LGF_LLM_API_KEY"] = key
            env["LGF_LLM_DEBUG"] = "1"
        else:
            env.pop("LGF_LLM_API_KEY", None)
        arm_log = os.path.join(REPO, "outputs", "asm", f"_v3_ablation_{tag}.log")
        log(f"[arm ] {tag} 启动 (backend={backend}, args={extra})")
        t0 = time.time()
        with open(arm_log, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)
        dt = (time.time() - t0) / 60
        if p.returncode != 0:
            log(f"[FAIL] {tag}: exit={p.returncode} after {dt:.1f} min, 详见 {arm_log}")
        else:
            log(f"[done] {tag}: {dt:.1f} min -> {done}")
    log(f"=== 全部完成, 总耗时 {(time.time()-t_all)/3600:.2f} h ===")


if __name__ == "__main__":
    main()
