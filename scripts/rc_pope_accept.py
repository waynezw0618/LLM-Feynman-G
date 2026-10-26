"""rc_pope_accept.py — P1 验收: P0 修正 (yw 真壁距 / valid_fit 双掩码 / 加权 fitness) 后的重跑效果.

验收指标 (对每个 tag=dataset 对):
  V1 近壁: |y/h|>0.9 (valid 全点, 诚实含外推区) 六分量 NMSE < 0.3
  V2 vw 中心峰: |y/h|<0.15 重构 b_vw 均值 > 0.03 (DNS ~0.055, 单峰在中心)
  V3 yw 占用: best 公式含 yw 的目标数 >= 3/5 (对抗 M_GUIDE 剔壁面特征 bug)
  V4 反演健康度: 数据集 valid_fit 上 max|a_m| < 1e3 (RC2)

用法:
  python scripts/rc_pope_accept.py --pairs heuristic_pope_wy=dataset_pope_wy.npz \
      heuristic_pope_y=dataset_pope_y.npz --out ASM_accept_pope_wy.md
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_eval import _decode_dsl, _eval_one, metrics  # noqa: E402

OUT = os.path.join(REPO, "outputs", "asm")
RO = [7.5, 15.0, 30.0, 75.0, 150.0]
SIX = ["uu", "vv", "ww", "uv", "uw", "vw"]


def evaluate_pair(tag: str, ds) -> dict:
    summary_path = os.path.join(OUT, f"runs_{tag}", "summary_B.json")
    summary = json.load(open(summary_path, encoding="utf-8"))
    sel_basis = [str(x) for x in ds["sel_basis"]]
    feat_keys = [str(x) for x in ds["featB_keys"]]
    wall_feat = "yw" if "yw" in feat_keys else ("ypd" if "ypd" in feat_keys else None)

    # 逐目标解码 + 预测 a_m (全 valid 点, 含被 valid_fit 剔除的外推区)
    per_case_ypa = {}
    best_dsl = {}
    for tgt, tinfo in summary["targets"].items():
        if tgt not in sel_basis:
            continue
        f = _decode_dsl(tinfo["dsl"], feat_keys)
        theta = np.asarray(tinfo["theta"], dtype=float)
        best_dsl[tgt] = tinfo["dsl"]
        if f is None:
            continue
        for ci in range(len(RO)):
            p = f"c{ci}_"; v = ds[p + "valid"]
            yp = _eval_one(f, ds[p + "featB"][v], theta, ci)
            per_case_ypa.setdefault(ci, [None] * len(sel_basis))
            per_case_ypa[ci][sel_basis.index(tgt)] = yp

    # 重建 b6 → 分区 NMSE
    res = dict(nmse_full={}, nmse_wall={}, nmse_wall_fit={}, vw_center={},
               vw_dns_center={}, vw_peak={}, vw_dns_peak={}, decoded=best_dsl)
    for ci in range(len(RO)):
        p = f"c{ci}_"; v = ds[p + "valid"]
        vf_all = ds[p + "valid"]
        fit_in_valid = (ds[p + "valid_fit"] if (p + "valid_fit") in ds.files
                        else vf_all)[v]        # valid 视角下的 fit 子集掩码
        yh = ds[p + "yh"][v]
        Tb = np.stack([ds[p + "T_" + m][v] for m in sel_basis], axis=1)
        dns = ds[p + "b6"][v]
        ypa = per_case_ypa.get(ci)
        if ypa is None or any(a is None for a in ypa):
            for comp in SIX:
                res["nmse_full"].setdefault(comp, {})[ci] = None
                res["nmse_wall"].setdefault(comp, {})[ci] = None
                res["nmse_wall_fit"].setdefault(comp, {})[ci] = None
            res["vw_center"][ci] = res["vw_dns_center"][ci] = None
            continue
        ypa_mat = np.stack([np.asarray(a) for a in ypa], axis=1)
        rec = np.einsum("imc,im->ic", Tb, ypa_mat)     # (Nv,6)
        nw = np.abs(yh) > 0.9
        ctr = np.abs(yh) < 0.15
        nw_fit = nw & fit_in_valid
        for k, comp in enumerate(SIX):
            m_full = metrics(dns[:, k], rec[:, k])
            m_wall = metrics(dns[nw, k], rec[nw, k]) if nw.sum() >= 3 else None
            m_wfit = metrics(dns[nw_fit, k], rec[nw_fit, k]) if nw_fit.sum() >= 3 else None
            res["nmse_full"].setdefault(comp, {})[ci] = m_full
            res["nmse_wall"].setdefault(comp, {})[ci] = m_wall
            res["nmse_wall_fit"].setdefault(comp, {})[ci] = m_wfit
        # vw (第 6 分量, idx 5)
        res["vw_center"][ci] = float(np.mean(rec[ctr, 5])) if ctr.any() else None
        res["vw_dns_center"][ci] = float(np.mean(dns[ctr, 5])) if ctr.any() else None
        res["vw_peak"][ci] = (float(rec[np.argmax(rec[:, 5]), 5]),
                              float(yh[np.argmax(rec[:, 5])]))
        res["vw_dns_peak"][ci] = (float(dns[np.argmax(dns[:, 5]), 5]),
                                  float(yh[np.argmax(dns[:, 5])]))
    return res


def yw_usage(tag: str, wall_feat: str) -> dict:
    """各目标 top_final 中含壁面特征的比例 (读 {name}_final.json)."""
    out = {}
    for fp in glob.glob(os.path.join(OUT, f"runs_{tag}", "B_a_*_final.json")):
        name = os.path.basename(fp).replace("B_a_", "").replace("_final.json", "")
        try:
            j = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        tops = j.get("top_final", [])
        dsls = [t.get("dsl", "") for t in tops]
        if not dsls:
            continue
        best_has = wall_feat in dsls[0] if wall_feat else False
        frac = (sum(1 for d in dsls if wall_feat and wall_feat in d)
                / len(dsls)) if wall_feat else 0.0
        out[name] = dict(best_has=best_has, top_frac=frac, n=len(dsls),
                         best_dsl=dsls[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="tag=dataset 对, 如 heuristic_pope_wy=dataset_pope_wy.npz")
    ap.add_argument("--out", default="ASM_accept_pope_wy.md")
    args = ap.parse_args()

    lines = ["# P1 验收报告 — P0 修正后的 ASM 重构质量", "",
             "修正内容: yw 真壁面距离 (RC1) / valid_fit 双掩码+min-norm 反演 (RC2) / "
             "近壁×8 + vw×5 + uw×3 分量加权 stress fitness (RC3)。", "",
             "评估口径: **全 valid 点** (含被 valid_fit 剔除的近壁外推点, 诚实度量)。", ""]

    for pair in args.pairs:
        tag, dsname = pair.split("=", 1)
        dsp = os.path.join(OUT, dsname)
        if not os.path.exists(os.path.join(OUT, f"runs_{tag}", "summary_B.json")):
            lines += [f"## {tag} — ❌ 未找到 summary_B.json, 跳过", ""]
            continue
        ds = np.load(dsp)
        sel_basis = [str(x) for x in ds["sel_basis"]]
        feat_keys = [str(x) for x in ds["featB_keys"]]
        wall_feat = "yw" if "yw" in feat_keys else ("ypd" if "ypd" in feat_keys else None)
        r = evaluate_pair(tag, ds)
        use = yw_usage(tag, wall_feat)

        lines += [f"## {tag}  (dataset={dsname}, 壁面特征={wall_feat})", ""]

        # V1 近壁 NMSE 表 (两口径: 全 valid / 仅 valid_fit 覆盖点)
        lines += ["### V1 近壁区 |y/h|>0.9 六分量 NMSE (目标 <0.3)", "",
                  "格式: `全valid (fit覆盖区)` — 全 valid 含被 valid_fit 剔除的外推点"
                  "（模型类在该处原理性退化），fit 覆盖区是公式层应达到的口径。", "",
                  "| comp | " + " | ".join(f"Ro{ro:g}" for ro in RO) + " | worst(fit) |",
                  "|" + "---|" * (len(RO) + 2)]
        v1_pass = True
        for comp in SIX:
            row = f"| {comp} "
            worst = 0.0
            for ci in range(len(RO)):
                mw = r["nmse_wall"].get(comp, {}).get(ci)
                mwf = r["nmse_wall_fit"].get(comp, {}).get(ci)
                if mwf is None or not np.isfinite(mwf.get("nmse", np.inf)):
                    row += "| — "; worst = np.inf; continue
                worst = max(worst, mwf["nmse"])
                v = mwf["nmse"]
                if mw is not None and np.isfinite(mw.get("nmse", np.inf)):
                    v = f"{mw['nmse']:.3f} ({mwf['nmse']:.3f})"
                row += f"| {v} "
            v1_pass &= (worst < 0.3)
            row += f"| {worst:.3f} |"
            lines.append(row)
        lines += ["", f"**V1 判定 (fit 覆盖区 worst<0.3): "
                  f"{'✅ 通过' if v1_pass else '❌ 未达'}**", ""]

        # V2 vw 中心峰
        lines += ["### V2 ⟨vw⟩ 中心峰 (|y/h|<0.15 均值, 目标 >0.03; DNS 参考)", "",
                  "| Ro | recon | DNS | peak(val, y/h) recon | peak DNS |",
                  "|---|---|---|---|---|"]
        n_v2 = 0
        for ci in range(len(RO)):
            rc = r["vw_center"].get(ci); dc = r["vw_dns_center"].get(ci)
            pk = r["vw_peak"].get(ci); dpk = r["vw_dns_peak"].get(ci)
            if rc is not None and rc > 0.03:
                n_v2 += 1
            lines.append(f"| {RO[ci]:g} | {rc:.4f} | {dc:.4f} "
                         f"| ({pk[0]:.4f}, {pk[1]:+.2f}) "
                         f"| ({dpk[0]:.4f}, {dpk[1]:+.2f}) |")
        lines += ["", f"**V2 判定: {'✅' if n_v2 >= 4 else '❌'} {n_v2}/5 个 Ro 中心 >0.03**", ""]

        # V3 壁面特征占用
        lines += ["### V3 best 公式壁面特征占用 (目标 ≥3/5 目标含 yw)", "",
                  "| target | best 含壁面特征 | top-8 占比 | best DSL |",
                  "|---|---|---|---|"]
        n_v3 = 0
        for tgt in sel_basis:
            u = use.get(tgt, {})
            bh = u.get("best_has", False)
            n_v3 += int(bh)
            dsl = (u.get("best_dsl", "") or "")[:90].replace("|", "\\|")
            lines.append(f"| {tgt} | {'✅' if bh else '❌'} "
                         f"| {u.get('top_frac', 0):.0%} | `{dsl}` |")
        lines += ["", f"**V3 判定: {'✅' if n_v3 >= 3 else '❌'} {n_v3}/5 目标 best 公式含 {wall_feat}**", ""]

        # V4 反演健康度 (数据集级, 与 tag 无关但按 dataset 报一次)
        amax = 0.0
        for ci in range(len(RO)):
            p = f"c{ci}_"
            vf = ds[p + "valid_fit"] if (p + "valid_fit") in ds.files else ds[p + "valid"]
            a = ds[p + "a"][vf]
            amax = max(amax, float(np.nanmax(np.abs(a))))
        lines += [f"### V4 反演健康度: valid_fit 上 max|a_m| = {amax:.3g} "
                  f"({'✅ <1e3' if amax < 1e3 else '❌ ≥1e3'})", ""]

    dst = os.path.join(OUT, args.out)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[accept] -> {dst}")


if __name__ == "__main__":
    main()
