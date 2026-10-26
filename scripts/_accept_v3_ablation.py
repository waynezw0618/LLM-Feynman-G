# -*- coding: utf-8 -*-
"""v3 消融验收: 8 臂 ({heur,mm}x{bare,full}x{mg,go}) x 5 目标 对比表.

输出:
  V1  5 目标 loss 矩阵 (8 臂) + 各配置内 heur vs mm 配对对比 (主判据)
  V2  M_GUIDE 通道统计 (n_calls/nonempty/applied/alive) — 通道全开验证
  V3  yw 占用 (best 是否含 yw) 逐臂逐目标
  V4  mg vs go 净效应 (同后端同配置内差值)

用法: python -X utf8 scripts/_accept_v3_ablation.py
缺臂时自动跳过 (跑批中途可随时跑, 看已有结果).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASM = os.path.join(REPO, "outputs", "asm")
TARGETS = ["T1", "T2v", "T3v", "T4v", "T5g"]
ARMS = ["v3_heur_bare_mg", "v3_mm_bare_mg", "v3_heur_bare_go", "v3_mm_bare_go",
        "v3_heur_full_mg", "v3_mm_full_mg", "v3_heur_full_go", "v3_mm_full_go"]


def load(tag):
    p = os.path.join(ASM, f"runs_{tag}", "summary_B.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def best_has_yw(dsl, ds):
    try:
        keys = [str(x) for x in ds["featB_keys"]]
        return "yw" in keys and "yw" in dsl
    except Exception:
        return "yw" in dsl


def main():
    import numpy as np
    ds = np.load(os.path.join(ASM, "dataset_pope_gsonly.npz"))
    data = {a: load(a) for a in ARMS}
    have = [a for a in ARMS if data[a]]
    if not have:
        sys.exit("尚无任何 summary_B.json")

    print("=" * 100)
    print("v3 消融验收: 2x2x2 = {heur, MiniMax-M2.7} x {bare, full} x {M_GUIDE on, ginit-only}")
    print(f"已完成臂: {len(have)}/8 -> {have}")
    print("=" * 100)

    # ---- V1: loss 矩阵 ----
    print("\n## V1: 5 目标 loss (各臂 fitness 口径见备注; bare=coef, full=stress)")
    print("  T        " + "".join(f"{a.replace('v3_',''):>18}" for a in ARMS))
    for t in TARGETS:
        row = f"  {t:<8}"
        for a in ARMS:
            d = data[a]
            v = d["targets"][t]["loss"] if d and t in d.get("targets", {}) else None
            row += f"{v:>18.4f}" if isinstance(v, (int, float)) else f"{'--':>18}"
        print(row)
    row = f"  {'mean':<8}"
    for a in ARMS:
        d = data[a]
        vs = [d["targets"][t]["loss"] for t in TARGETS
              if d and t in d.get("targets", {})]
        row += f"{(sum(vs)/len(vs)):>18.4f}" if vs else f"{'--':>18}"
    print(row)
    print("  (注意: bare 与 full 之间 loss 不可直接比 — fitness 口径不同; "
          "只做同配置内 heur vs mm 对比)")

    # ---- 配对对比 (主判据) ----
    for cfg_name in ("bare", "full"):
        h, m = f"v3_heur_{cfg_name}_mg", f"v3_mm_{cfg_name}_mg"
        if data.get(h) and data.get(m):
            print(f"\n## {cfg_name.upper()} 对决 (通道全开, 主判据): heur vs M2.7")
            print("  T        heur       mm27     mm/heur")
            diffs = []
            for t in TARGETS:
                lh = data[h]["targets"].get(t, {}).get("loss")
                lm = data[m]["targets"].get(t, {}).get("loss")
                if lh is not None and lm is not None:
                    r = lm / lh if lh else float("inf")
                    diffs.append(r - 1.0)
                    print(f"  {t:<8}{lh:>9.4f}{lm:>11.4f}{r:>10.1%}")
            if diffs:
                print(f"  平均: M2.7 相对 heur {'+' if sum(diffs)>0 else ''}"
                      f"{sum(diffs)/len(diffs):.1%} (负=M2.7 胜)")

    # ---- V2: M_GUIDE 通道 ----
    print("\n## V2: M_GUIDE 通道统计 (calls/nonempty/applied/alive; go 臂应为 0)")
    print("  arm                        calls  nonempty  applied  alive  last_err")
    for a in ARMS:
        d = data[a]
        if not d:
            continue
        mg = None
        for t in TARGETS:
            st = d["targets"].get(t, {}).get("llm_stats", {}).get("mguide")
            if st:
                mg = st if mg is None else {
                    k: (mg.get(k, 0) + st.get(k, 0)
                        if isinstance(st.get(k), (int, float)) else mg.get(k))
                    for k in set(mg) | set(st)}
        if mg:
            print(f"  {a.replace('v3_',''):<26}{mg.get('n_calls',0):>6}{mg.get('n_nonempty',0):>10}"
                  f"{mg.get('n_applied',0):>9}{mg.get('n_alive',0):>7}"
                  f"  {'Y' if mg.get('last_error') or mg.get('last_exception') else '-'}")

    # ---- V3: yw 占用 ----
    print("\n## V3: yw 占用 (best 公式含 yw 的目标数 / 已完成目标数)")
    for a in ARMS:
        d = data[a]
        if not d:
            continue
        ys = "".join("Y" if best_has_yw(d["targets"][t]["dsl"], ds) else "."
                     for t in TARGETS if t in d.get("targets", {}))
        print(f"  {a.replace('v3_',''):<26} {ys}")

    # ---- V4: mg vs go 净效应 ----
    print("\n## V4: M_GUIDE 净效应 (go 臂 loss - mg 臂 loss; 正 = M_GUIDE 有益)")
    for be in ("heur", "mm"):
        for cfg_name in ("bare", "full"):
            g1, g0 = f"v3_{be}_{cfg_name}_mg", f"v3_{be}_{cfg_name}_go"
            if data.get(g1) and data.get(g0):
                print(f"  {be}_{cfg_name}: ", end="")
                for t in TARGETS:
                    l1 = data[g1]["targets"].get(t, {}).get("loss")
                    l0 = data[g0]["targets"].get(t, {}).get("loss")
                    if l1 is not None and l0 is not None:
                        print(f"{t}={l0-l1:+.4f} ", end="")
                print()


if __name__ == "__main__":
    main()
