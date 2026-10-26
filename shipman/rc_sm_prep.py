"""shipman.rc_sm_prep — 阶段1: 合成数据落盘.

用法:
  python -X utf8 shipman/rc_sm_prep.py [--ships kvlcc2] [--seeds 0] [--cal_frac 0.10]

输出: outputs/shipman/sim_{ship}_{seed}.npz
  键: c{i}_{u,v,r,delta,psi,x,y, XH,YH,NH, XP,XR,YR,NR, Ye_ext,Ne_ext,
      udot,vdot,rdot, yY,yN,yX} + meta (ship, seed, noise list, 扩展真值系数).
c{i} 前缀对齐 asm npz 约定; i = 机动序号 (0=turn35p 1=turn35s 2=zig1010 3=zig2020).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from shipman.coeff_kvlcc2 import kvlcc2          # noqa: E402
from shipman.mmg_sim import simulate               # noqa: E402
from shipman.variants import calibrate_ext, graybox_targets  # noqa: E402

OUT_DIR = os.path.join(REPO, "outputs", "shipman")
MANEUVERS = ("turn35p", "turn35s", "zig1010", "zig2020")


def make_ship(ship: str, seed: int, cal_frac: float):
    if ship == "kvlcc2":
        return kvlcc2()
    raise ValueError(f"未知船型 {ship} (kcs 系数待 M3 落地)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ships", default="kvlcc2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cal_frac", type=float, default=0.10,
                    help="扩展真值新项的目标力占比 (V2/V3)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ship in args.ships.split(","):
        for seed in [int(x) for x in args.seeds.split(",")]:
            s0 = make_ship(ship, seed, args.cal_frac)
            # 标定扩展真值 (V2: cDv/cDr @α=0.5; V3: mem_A)
            s2 = calibrate_ext(s0, target_frac=args.cal_frac, seed=seed)
            out_path = os.path.join(OUT_DIR, f"sim_{ship}_{seed}.npz")
            save = {
                "meta_ship": ship, "meta_seed": seed,
                "meta_cal_frac": args.cal_frac,
                "meta_cDv": s2.cDv, "meta_cDr": s2.cDr,
                "meta_alpha": s2.alpha_v,
                "meta_mem_A": np.asarray(s2.mem_A, dtype=float).reshape(-1),
            }
            for i, mn in enumerate(MANEUVERS):
                # V1 基准数据 (无扩展项) + V2/V3 扩展数据 各存一份
                for tag, s_used in (("v1", s0), ("v2", s2)):
                    # v2 数据用扩展模型生成; v3 与 v2 共用 (mem 已含于 s2)
                    if tag == "v1" and s0 is s2:
                        pass
                    out = simulate(s_used, mn, seed=seed)
                    tg = graybox_targets(s_used, out)
                    pfx = f"c{i}_{tag}_"
                    for k in ("t", "u", "v", "r", "delta", "psi", "x", "y",
                              "XH", "YH", "NH", "XP", "XR", "YR", "NR",
                              "Ye_ext", "Ne_ext", "udot", "vdot", "rdot"):
                        save[pfx + k] = out[k]
                    save[pfx + "yY"] = tg["yY"]
                    save[pfx + "yN"] = tg["yN"]
                    save[pfx + "yX"] = tg["yX"]
            np.savez_compressed(out_path, **save)
            print(f"[prep] {out_path} ({len(save)} keys, cDv={s2.cDv:.5f}, "
                  f"cDr={s2.cDr:.5f}, memA={s2.mem_A})")


if __name__ == "__main__":
    main()
