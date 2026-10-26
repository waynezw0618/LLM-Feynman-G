# -*- coding: utf-8 -*-
"""rc_asm_true_prep.py — 真 ASM (true Algebraic Stress Model) 数据管线 v2.

参考 Weatheritt & Sandberg (2016 JCP 325:22-37) 式(17)-(20) 的结构:
  a_ij = sum_n f_n(...) * V^n_ij  — 单一闭式标量函数 f, 对全部 6 个各向异性
  分量与全部 5 个旋转工况统一 (只允许全局常数 C#, 无逐工况参数 K#).

v2 (2026-10-09, 用户两条修正):
  1. 张量基不限于 eq.(18) 的 4 基最小集 — 输入 Pope(1975) 全部 10 个
     对称无迹张量基 T1..T10, 由 SR 方法 (PySR / LGF) 自行选择;
  2. 标量不变量放开到全部 5 个 (eq.18 只有 I1, I2): eta2, zeta2, s3, so2, s2o2;
  3. 模型形式显式包含 Re_t 与 Ro_loc (v1 缺失/不完整).

模型形式 (17 特征; 行 = (工况, 壁面位置, 张量分量) 三元组):
  b_ij = f( T1_ij..T10_ij, eta2, zeta2, s3, so2, s2o2, ret, roloc )
  x1..x10  = 该行的 (ij) 分量在 10 个基张量上的取值 (S, W 均乘 k/eps 归一化;
             与 W&S 的 2*beta*k/eps 归一化只差常数因子, 由系数吸收)
  x11..x15 = eta2=tr(S^2), zeta2=-tr(W^2), s3=tr(S^3), so2=tr(SW^2), s2o2=tr(S^2W^2)
  x16      = ret = k^2/(nu*eps), nu=1/Re_tau=1/180 (局部湍流雷诺数, 无量纲)
  x17      = roloc = 2*Om_eff*k/eps (带符号局部旋转数; 跨工况差异的主要载体)

涡量变体 (用户指定三种):
  v: W = Wbar              (纯涡量; 帧旋转仅经 roloc 进入)
  g: W = Wbar + 1*eps*Om   (Gatski-Speziale 1993)
  e: W = Wbar + 2*eps*Om   (扩展内禀)
注: Om_[:,1,2] 为 0 基索引 = 张量 (2,3) 分量 — 流向旋转轴 x1 的 eps*Omega
    只有 23/32 分量, 与 rc_pope_prep.py 的 Wg/We 构造逐字一致.

与既有 coefSR (Pope 5 基逐目标系数回归) 的本质区别: 同一公式必须跨
6 分量 x 5 工况通用; 常数只有全局 C#; 评估加 rho_a,case (W&S eq.22).

输出 outputs/asm/dataset_trueasm_{v,g,e}.npz:
  feat_names (17,) = [T1..T10, eta2, zeta2, s3, so2, s2o2, ret, roloc]
  c{i}_T{Tm} (N,6) x10 全网格基张量六分量 [11,22,33,12,13,23]
  c{i}_b6/yh/k/eps/valid/ro;  c{i}_scalars (N,7)
  c{i}_Xfit (6Nv,17), c{i}_yfit (6Nv,) — 仅 valid 行, 分量优先 row=c*Nv+j
参考基线 (打印 + prep_report_trueasm.json):
  linear  : b ~ c*T1 全局单常数 (线性涡黏闭包对照)
  cc10    : 10 基全局常数系数 (经典 Pope TBFCM 常数版, 5 工况联合, 10 常数)
  cc10_pc : 逐工况独立常数 (工况特异性上界参考, 5x10 常数, 非闭式)
  oracle4 : T1..T4 逐点 lstsq (4 基线性张成上限, 逐点系数任意)
  oracle10: T1..T10 逐点 lstsq (10 基张成上限; 6 维分量空间被张满则 ~0)

用法: python -X utf8 scripts/rc_asm_true_prep.py [--variants v g e]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rc_asm_prep import load_case, sym6, RO_CASES, OUT_DIR  # noqa: E402
from rc_pope_prep import pope_tensors, invariants  # noqa: E402

RE_TAU = 180.0
TEN = [f"T{m}" for m in range(1, 11)]
SCAL_NAMES = ["eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc"]
FEAT_NAMES = TEN + SCAL_NAMES
VARIANTS = {
    "v": dict(rot_factor=0.0, label="vorticity: W = Wbar (frame rotation only via roloc)"),
    "g": dict(rot_factor=1.0, label="Gatski-Speziale 1993: W = Wbar + 1*eps*Om"),
    "e": dict(rot_factor=2.0, label="extended intrinsic: W = Wbar + 2*eps*Om"),
}


def rows_of(M6: np.ndarray) -> np.ndarray:
    """(N,6) 分量矩阵 -> (6N,) 分量优先行展开: index = c*N + j."""
    return M6.T.reshape(-1)


def build_variant(vkey: str) -> tuple[dict, dict]:
    rf = VARIANTS[vkey]["rot_factor"]
    save = {"feat_names": np.array(FEAT_NAMES), "variant": np.array(vkey),
            "variant_label": np.array(VARIANTS[vkey]["label"]),
            "ref": np.array("Weatheritt & Sandberg 2016 JCP 325:22-37 eq.(17)-(20); "
                            "v2: full Pope T1..T10 + 5 invariants + Re_t/Ro_loc; "
                            "grad normalized by k/eps; universal constants only")}
    meta = dict(variant=vkey, label=VARIANTS[vkey]["label"], cases=[])
    lin_xy = lin_xx = lin_yy = 0.0            # linear 基线累积
    Xall, yall = [], []                       # cc10 全局常数基线 (跨工况联合)
    ccpc_sse = ccpc_ssb = 0.0                  # cc10 逐工况基线累积
    o4_sse = o10_sse = ssb = 0.0               # oracle 累积
    Tnorm = {n: 0.0 for n in TEN}              # 基幅值画像
    ranks10 = []
    for i, ro in enumerate(RO_CASES):
        fc = load_case(ro, variant="vort")     # Ot 恒为纯涡量; 帧旋转项手动加
        ke = (fc["k"] / fc["eps"])[:, None, None]
        Ss = ke * fc["S"]
        Om_ = fc["Ot"].copy()
        if rf != 0.0:
            Om_[:, 1, 2] += rf * fc["Om_eff"]
            Om_[:, 2, 1] -= rf * fc["Om_eff"]
        Ws = ke * Om_
        Tt = dict(pope_tensors(Ss, Ws))        # {T1..T10: (N,3,3)}
        inv = invariants(Ss, Ws)
        ret = RE_TAU * fc["k"] ** 2 / fc["eps"]     # k^2/(nu*eps), nu=1/180
        roloc = 2.0 * fc["Om_eff"] * fc["k"] / fc["eps"]
        b6 = fc["b6"]
        N = b6.shape[0]
        T6_ = {n: sym6(T) for n, T in Tt.items()}
        scal = np.stack([inv["eta2"], inv["zeta2"], inv["s3"], inv["so2"],
                         inv["s2o2"], ret, roloc], axis=1)          # (N,7)
        valid = ((fc["k"] > 1e-4) & (fc["eps"] > 1e-12)
                 & np.isfinite(b6).all(1) & np.isfinite(scal).all(1)
                 & np.isfinite(np.concatenate([T6_[n] for n in TEN], 1)).all(1))
        # ---- 拟合行 (仅 valid; 分量优先) ----
        Tv = {n: T6_[n][valid] for n in TEN}
        Nv = int(valid.sum())
        Xfit = np.column_stack(
            [rows_of(Tv[n]) for n in TEN]
            + [np.tile(scal[valid][:, j], 6) for j in range(7)])
        yfit = rows_of(b6[valid])
        # ---- 基线: linear (b = c*T1, 全局单常数) ----
        x1 = Xfit[:, 0]
        lin_xy += float(np.dot(x1, yfit)); lin_xx += float(np.dot(x1, x1))
        lin_yy += float(np.dot(yfit, yfit))
        # ---- 基线: cc10 (全局 10 常数) — 累积, 循环后联合拟合; cc10_pc (逐工况) ----
        Xall.append(Xfit[:, :10]); yall.append(yfit)
        c_pc, *_ = np.linalg.lstsq(Xfit[:, :10], yfit, rcond=1e-12)  # 本 case 独立
        r_pc = Xfit[:, :10] @ c_pc - yfit
        ccpc_sse += float(np.dot(r_pc, r_pc)); ccpc_ssb += float(np.dot(yfit, yfit))
        # ---- oracle4 / oracle10 (逐点 lstsq, 系数任意) ----
        M4 = np.stack([T6_[n] for n in TEN[:4]], axis=2)   # (N,6,4)
        M10 = np.stack([T6_[n] for n in TEN], axis=2)      # (N,6,10)
        for j in np.where(valid)[0]:
            s4, *_ = np.linalg.lstsq(M4[j], b6[j], rcond=1e-10)
            o4_sse += float(np.sum((M4[j] @ s4 - b6[j]) ** 2))
            sv = np.linalg.svd(M10[j], compute_uv=False)
            rk = int(np.sum(sv / max(sv[0], 1e-300) > 1e-10))
            ranks10.append(min(rk, 6))
            if rk >= 6:
                s10, *_ = np.linalg.lstsq(M10[j], b6[j], rcond=1e-10)
                o10_sse += 0.0
            else:
                s10, *_ = np.linalg.lstsq(M10[j], b6[j], rcond=1e-10)
                o10_sse += float(np.sum((M10[j] @ s10 - b6[j]) ** 2))
        ssb += float(np.sum(b6[valid] ** 2))
        for n in TEN:
            Tnorm[n] = max(Tnorm[n], float(np.abs(T6_[n][valid]).max()))
        # ---- 落盘 ----
        p = f"c{i}_"
        save.update({
            p + "ro": fc["ro"], p + "yh": fc["yh"], p + "k": fc["k"],
            p + "eps": fc["eps"], p + "valid": valid, p + "b6": b6,
            p + "scalars": scal, p + "Xfit": Xfit, p + "yfit": yfit,
        })
        for n in TEN:
            save[p + n] = T6_[n]
        meta["cases"].append(dict(
            ro=float(ro), n_pts=int(N), n_valid=Nv, n_rows=int(6 * Nv),
            scalar_ranges={k: [float(scal[valid][:, j].min()),
                               float(scal[valid][:, j].max())]
                           for j, k in enumerate(SCAL_NAMES)},
            b6_absmax=float(np.abs(b6[valid]).max())))
        print(f"[prep:{vkey}] Ro={ro:>5}: {N}pts valid={Nv} rows={6*Nv} | "
              + " ".join(f"{k}[{scal[valid][:,j].min():.3g},{scal[valid][:,j].max():.3g}]"
                         for j, k in enumerate(SCAL_NAMES)))
    c_lin = lin_xy / max(lin_xx, 1e-300)
    # ---- cc10 全局常数: 跨工况联合 lstsq (5 工况共用 10 常数) ----
    Xg = np.vstack(Xall); yg = np.concatenate(yall)
    c_g, *_ = np.linalg.lstsq(Xg, yg, rcond=1e-12)
    r_g = Xg @ c_g - yg
    cc_sse = float(np.dot(r_g, r_g)); cc_ssb = float(np.dot(yg, yg))
    meta["baselines"] = dict(
        linear=dict(c=float(c_lin),
                    rel_resid=float(np.sqrt(max(
                        lin_yy - 2 * c_lin * lin_xy + c_lin * c_lin * lin_xx, 0.0)
                        / max(lin_yy, 1e-300))),
                    note="b ~ c*T1, global single constant"),
        cc10=dict(rel_resid=float(np.sqrt(cc_sse / max(cc_ssb, 1e-300))),
                  coeffs=[float(v) for v in c_g],
                  note="10-basis global constant coefficients (Pope TBFCM, 10 consts, "
                       "joint lstsq over all 5 cases)"),
        cc10_pc=dict(rel_resid=float(np.sqrt(ccpc_sse / max(ccpc_ssb, 1e-300))),
                     note="per-case independent constants (50 consts, non-universal)"),
        oracle4=dict(rel_resid=float(np.sqrt(o4_sse / max(ssb, 1e-300))),
                     note="per-point lstsq on T1..T4"),
        oracle10=dict(rel_resid=float(np.sqrt(o10_sse / max(ssb, 1e-300))),
                      note="per-point lstsq on T1..T10"))
    meta["basis_absmax"] = Tnorm
    meta["rank10_hist"] = {str(r): int(np.sum(np.array(ranks10) == r))
                           for r in range(7)}
    print(f"[prep:{vkey}] linear c={c_lin:.4f} | cc10={meta['baselines']['cc10']['rel_resid']:.4f}"
          f" | cc10_pc={meta['baselines']['cc10_pc']['rel_resid']:.4f}"
          f" | oracle4={meta['baselines']['oracle4']['rel_resid']:.4f}"
          f" | oracle10={meta['baselines']['oracle10']['rel_resid']:.4f}"
          f" | rank10={meta['rank10_hist']}")
    return save, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["v", "g", "e"],
                    choices=list(VARIANTS))
    args = ap.parse_args()
    metas = {}
    for vkey in args.variants:
        save, meta = build_variant(vkey)
        out = os.path.join(OUT_DIR, f"dataset_trueasm_{vkey}.npz")
        np.savez_compressed(out, **save)
        metas[vkey] = meta
        print(f"[prep:{vkey}] -> {out}")
    with open(os.path.join(OUT_DIR, "prep_report_trueasm.json"), "w",
              encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print("[prep] meta -> prep_report_trueasm.json")


if __name__ == "__main__":
    main()
