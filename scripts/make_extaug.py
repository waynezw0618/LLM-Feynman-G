"""make_extaug.py — 在 dataset.npz (ext 变体) 基础上追加增广特征列, 生成 dataset_extaug.npz.

动机: a_S / a_S2 在 5 不变量特征集下 R2 很低 (a_S ~ 0), 怀疑特征表达力不足.
增广特征 (打破不变量约束, 属于探索性建模, 不可外推):
  logke  = log(k/eps)      湍流时间尺度 (近壁 log 区特征)
  yplus  = y+              无量纲壁距 (直接编码近壁结构)

用法: python scripts/make_extaug.py
输出: outputs/asm/dataset_extaug.npz
"""
from __future__ import annotations
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "asm")
N_CASES = 5
EXTRA = ["logke", "yplus"]


def main():
    src = os.path.join(OUT, "dataset.npz")
    dst = os.path.join(OUT, "dataset_extaug.npz")
    ds = np.load(src)
    save = {k: ds[k] for k in ds.files}

    keys = [str(x) for x in ds["featB_keys"]]
    featB_keys = keys + EXTRA
    save["featB_keys"] = np.array(featB_keys)
    for i in range(N_CASES):
        p = f"c{i}_"
        k = ds[p + "k"]; eps = ds[p + "eps"]; yp = ds[p + "yp"]
        logke = np.log(np.maximum(k / np.maximum(eps, 1e-300), 1e-12))
        extra = np.stack([logke, yp], axis=1)
        save[p + "featB"] = np.concatenate([ds[p + "featB"], extra], axis=1)
        print(f"[extaug] c{i}: logke range [{logke.min():.2f}, {logke.max():.2f}], "
              f"yplus range [{yp.min():.1f}, {yp.max():.1f}]")

    np.savez_compressed(dst, **save)
    print(f"[extaug] featB_keys = {featB_keys}")
    print(f"[extaug] -> {dst}")


if __name__ == "__main__":
    main()
