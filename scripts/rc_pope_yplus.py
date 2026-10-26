"""rc_pope_yplus.py — Pope 数据集 + 真壁面距离特征 yw = (1-|y/h|)*Re_tau.

⚠ 2026-09-30 修正 (RC1): 原实现 ypd = |y+| = |y/h|*Re_tau, 而 d[:,0]=y/h 从
**槽道中心**起算 (壁面 y/h=±1 ↔ y+ = ±180) → 原 ypd 是"离中心距离", 不是
壁面距离! van Driest 型 φ(ypd) 实际阻尼中心、释放壁面 — 方向反了.

本版: yw = (1-|y/h|)*Re_tau = 180-|y+|, 真实到最近壁面的无量纲距离
(wall units). yw=0 在壁面, yw=180 在槽道中心. 近壁衰减函数 φ(yw) 的
van Driest 语义恢复正确: φ→0 (壁面), φ→1 (远离壁面).

DSL 可表达的 φ(yw) 形状 (add/sub/mul/div/exp/log/tanh/sqrt/inv/neg):
  1 - exp(-C*yw)        (van Driest, 壁面阻尼)
  yw/(yw + C)           (有理式阻尼, 壁面→0)
  tanh(C*yw)            (饱和型)
  exp(-C/yw) 等

用法: python scripts/rc_pope_yplus.py [--src dataset_pope2.npz] [--out dataset_pope_wy.npz]
"""
from __future__ import annotations
import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "outputs", "asm")

# 真壁面距离 (wall units), 放在特征列末尾, 名字 "yw"
FEAT_ADD = "yw"
RE_TAU = 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_pope2.npz")
    ap.add_argument("--out", default="dataset_pope_wy.npz")
    args = ap.parse_args()

    src = dict(np.load(os.path.join(OUT_DIR, args.src)))
    kb = [str(x) for x in src["featB_keys"]]
    ka = [str(x) for x in src["featA_keys"]]
    assert FEAT_ADD not in kb, f"{FEAT_ADD} 已存在"
    out = dict(src)
    out["featB_keys"] = np.array(kb + [FEAT_ADD])
    out["featA_keys"] = np.array(ka + [FEAT_ADD])
    for i in range(5):
        p = f"c{i}_"
        yh = src[p + "yh"]
        v = src[p + "valid"]
        # 真壁面距离: 壁面 y/h=±1 → yw=0; 中心 y/h=0 → yw=Re_tau
        yw = ((1.0 - np.abs(yh)) * RE_TAU).astype(src[p + "featB"].dtype)
        out[p + "featB"] = np.concatenate([src[p + "featB"], yw[:, None]], axis=1)
        out[p + "featA"] = np.concatenate([src[p + "featA"], yw[:, None]], axis=1)
        print(f"[yplus] Ro={src[p + 'ro']:>5g}: yw range (valid) "
              f"[{yw[v].min():.1f}, {yw[v].max():.1f}]  "
              f"(yw=0 壁面, yw=180 中心)")
    # 校验: 形状/键一致
    assert out["c0_featB"].shape[1] == len(out["featB_keys"])
    dst = os.path.join(OUT_DIR, args.out)
    np.savez_compressed(dst, **out)
    print(f"[yplus] -> {dst}  (featB {len(kb)}+{1} 列: {kb + [FEAT_ADD]})")


if __name__ == "__main__":
    main()
