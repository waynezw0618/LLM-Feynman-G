# -*- coding: utf-8 -*-
"""v3a 消融 launcher: 在 v3 的 8 臂结构上叠加 G_INIT 鲁棒性改进 (A1 解析重试 + A3 few-shot).

背景 (v3 已跑完, 2.50h):
  v3 裸 + G_INIT-only: mm 0.2231 vs heur 0.3425  → **M2.7 胜 -34.9%** (T5g 精确解 4e-5)
  v3 裸 + M_GUIDE   : mm 0.4487 vs heur 0.3212  → M2.7 输 +39.7% (T5g 被毁 0.611)
  → M_GUIDE 对 M2.7 是净负; G_INIT-only (B1) 才是 M2.7 的真实优势通道

v3a 变量 = 只叠加 A1+A3 (其余与 v3 逐项相同, 同 seed → 配对可比):
  A1: G_INIT 解析率不达标时重试 (--ginit_retry 2 --ginit_min_kept 3)
  A3: G_INIT prompt 注入 few-shot 合法 DSL 示例 (--ginit_fewshot 1)
  两臂同享 (非本消融变量): V3-0..d 管线修复 + G_INIT 硬规则

臂定义 (8 臂 x 5 目标 T1/T2v/T3v/T4v/T5g):
  bare = coef fitness + valid mask + 全默认权重 + 无 forbid + 无 wall-seeds
  full = stress fitness + valid_fit + wall_boost 8 + w_vw 5 + w_uw 3
         + forbid_map T5g:sub,div + B2 wall-seeds
  mg   = M_GUIDE 每 5 代中途干预开;  go = --ginit_only (LLM 仅出初始种群)

注意: heur 臂受 A1/A3 影响为零 (HeuristicLLM 忽略 prompt; 3 候选 >= min_kept) →
  预期与 v3 对应 heur 臂字节级一致, 充当 v3a 管线的可复现性对照.

臂序: mm 臂优先 (新信息), 各配置内先 go (v3 最优) 后 mg; 裸优先于 full.

用法: python -X utf8 scripts/_run_v3a_ablation.py
日志: outputs/asm/_v3a_ablation_progress.log + 每臂 _v3a_ablation_<tag>.log
"""
import os
import re
import subprocess
import sys
import time

REPO = r"d:\AI\LLM-Feynman-G"
CFG = r"C:\Users\lenovo\.config\evoscientist\config.yaml"
PY = sys.executable

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
# v3a 新变量: A1 重试 + A3 few-shot
A1A3 = ["--ginit_retry", "2", "--ginit_min_kept", "3", "--ginit_fewshot", "1"]

# 臂序: mm 优先, 裸优先, go 优先
ARMS = [
    ("v3a_mm_bare_go",   "minimax",   BARE + GO + A1A3),
    ("v3a_heur_bare_go", "heuristic", BARE + GO + A1A3),
    ("v3a_mm_bare_mg",   "minimax",   BARE + A1A3),
    ("v3a_heur_bare_mg", "heuristic", BARE + A1A3),
    ("v3a_mm_full_go",   "minimax",   FULL + GO + A1A3),
    ("v3a_heur_full_go", "heuristic", FULL + GO + A1A3),
    ("v3a_mm_full_mg",   "minimax",   FULL + A1A3),
    ("v3a_heur_full_mg", "heuristic", FULL + A1A3),
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
    with open(os.path.join(REPO, "outputs", "asm", "_v3a_ablation_progress.log"),
              "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    t_all = time.time()
    log(f"=== v3a 消融跑批启动: 8 臂 x 5 目标, 顺序 {[a[0] for a in ARMS]} ===")
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
        arm_log = os.path.join(REPO, "outputs", "asm", f"_v3a_ablation_{tag}.log")
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
