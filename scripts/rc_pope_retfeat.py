"""rc_pope_retfeat.py — Pope 数据集 + 局部湍流雷诺数 Re_t / 局部旋转数 Ro_local 特征.

新增两列 (粘性单位 ν=1):
  ret   = k^2 / eps            局部湍流雷诺数 (Pope Re_T; Myon-Kasagi/Launder-Sharma
                                低-Re 阻尼函数的标准变量)
  roloc = 2 * Om_eff * k/eps   局部旋转数 (Ω 为系统旋转, 取 load_case 判定的
                                带符号 Om_eff; 本流动族 5 工况同号)

实测形态 (dataset_pope_wy, 见 _ret_diag4.py):
  ret:   yw=1: ~0.02-0.07 → yw=10: 90-150 → 平台 188(Ro7.5)-317(Ro150)
         → y+∈[1,30] 跨 4 个量级, 恰是 V1 近壁失败区; 平台随 Ro 饱和
  roloc: 中心值 5.1(Ro7.5) → 127(Ro150)

⚠ 诚实声明: 工况内 Re_t 近平台常数 → 样本内 C(Ro) 与 C(Re_t) 几乎不可分,
  其跨工况统一价值需留一 Ro 外推实验检验. eps 继承 rc_asm_prep 耗散预算
  重建 (λ 修正) 的不确定性.

用法: python scripts/rc_pope_retfeat.py [--src dataset_pope_wy.npz]
          [--out dataset_pope_wy_rt.npz]
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_prep import load_case, RO_CASES, H_PLUS  # noqa: E402

OUT_DIR = os.path.join(REPO, "outputs", "asm")

FEATS_ADD = ["ret", "roloc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_pope_wy.npz")
    ap.add_argument("--out", default="dataset_pope_wy_rt.npz")
    args = ap.parse_args()

    src = dict(np.load(os.path.join(OUT_DIR, args.src)))
    kb = [str(x) for x in src["featB_keys"]]
    ka = [str(x) for x in src["featA_keys"]]
    for f in FEATS_ADD:
        assert f not in kb, f"{f} 已存在"
    out = dict(src)
    out["featB_keys"] = np.array(kb + FEATS_ADD)
    out["featA_keys"] = np.array(ka + FEATS_ADD)

    # Om_eff 需从原始数据重判符号 (npz 未存); load_case 轻量 (~129 行 txt)
    om_eff = {}
    for ro in RO_CASES:
        fc = load_case(ro, variant="vort")
        om_eff[ro] = fc["Om_eff"]
        print(f"[retfeat] Ro={ro:>5g}: Om_eff={fc['Om_eff']:+.5f} "
              f"(rot corr +{fc['checks']['rot_cpos']:.3f}/-{fc['checks']['rot_cneg']:.3f})")

    for i in range(5):
        p = f"c{i}_"
        ro = float(src[p + "ro"])
        k = src[p + "k"]
        eps = np.maximum(src[p + "eps"], 1e-30)
        v = src[p + "valid"]
        ret = (k ** 2 / eps).astype(src[p + "featB"].dtype)
        roloc = (2.0 * om_eff[ro] * k / eps).astype(src[p + "featB"].dtype)
        cols = np.stack([ret, roloc], axis=1)
        out[p + "featB"] = np.concatenate([src[p + "featB"], cols], axis=1)
        out[p + "featA"] = np.concatenate([src[p + "featA"], cols], axis=1)
        yh = src[p + "yh"]
        # 近壁 yh|>0.9 (yw<18) 的动态范围 — 阻尼变量的关键区间
        nb = v & (np.abs(yh) > 0.9)
        ctr = v & (np.abs(yh) < 0.15)
        print(f"[retfeat] Ro={ro:>5g}: ret(valid) [{ret[v].min():.3g},{ret[v].max():.3g}] "
              f"近壁[{ret[nb].min():.3g},{ret[nb].max():.3g}] 中心 {ret[ctr].mean():.1f} | "
              f"roloc(valid) [{roloc[v].min():.3g},{roloc[v].max():.3g}] "
              f"中心 {roloc[ctr].mean():.2f}")

    assert out["c0_featB"].shape[1] == len(out["featB_keys"])
    dst = os.path.join(OUT_DIR, args.out)
    np.savez_compressed(dst, **out)
    print(f"[retfeat] -> {dst}  (featB {len(kb)}+{len(FEATS_ADD)} 列: "
          f"{kb + FEATS_ADD})")


if __name__ == "__main__":
    main()
