# -*- coding: utf-8 -*-
"""rc_asm_tbfcm_var_mix.py — TBFCM 骨架: 变体可选 (v/g/e), 10 个 a_m 独立 LGF-go + 装配评估.

骨架 (TBFCM = Tensor-Basis Functional Coefficient Model):
    b_ij(I) = Σ_{m ∈ sel_basis} g_m(I) · T_m,ij(x)
其中:
    - g_m(I): 第 m 个基的标量系数函数, 输入 = 11 维不变量 (9 featB + ret + roloc)
    - T_m,ij: 张量基分量的 (x) 空间分布 (从 dataset_pope_wy 读)
    - sel_basis = [T1, T2v, T3v, T4v, T5v, T6v, T7v, T8v, T9v, T10v]  (扩展 10基)

训练 (per target m):
    损失 = mean_c ( Σ_j w_j · (ĝ_m - a_m)² / Σ_j w_j · (a_m - μ_w)² )
    其中 w_j = ||T_m,ij(x_j)||²_F   (stress fitness: 装配贡献加权)

护栏 (硬规则, 与方案 1+2 一致):
    forbidden_ops = ["tanh"]     (大增益饱和 → 平坦区, 整条丢弃)
    global_params_only = True    (禁 K#, 全局 C# 共享所有工况)
    llm_ginit_only = True        (M_GUIDE 已证实失效, 只用 G_INIT 一次)

后端: minimax (MiniMax-M2.7) 默认, 5 个目标串行 (每个 ~5-7 min)

输出:
    outputs/asm/runs_tbfcm_mix_lgf_go/{m}/<m>_final.json  (per-target)
    outputs/asm/runs_tbfcm_mix_lgf_go/tbfcm_result.json   (装配 + 评估)
    outputs/asm/ASM_tbfcm_mix_report.md                   (Markdown 报告)
    outputs/asm/figs_trueasm/fig_tbfcm_*.png              (剖面 + 装配对比图)

用法:
    python -X utf8 scripts/rc_asm_tbfcm_mix.py
    python -X utf8 scripts/rc_asm_tbfcm_mix.py --targets T2v T1
    python -X utf8 scripts/rc_asm_tbfcm_mix.py --skip_sr
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rc_asm_run import BACKENDS, RO_CASES
from rc_asm_true_eval import (eval_predictions, summarize, COMP6,
                              _full_tensor, rho_a_profile)
from lgf.config import LGFConfig
from lgf import llm_ops, dsl as lgfdsl
from lgf.compiler import CompiledGraph
from lgf.pipeline_lgf import run_lgf

OUT_DIR = REPO / "outputs" / "asm"
RUN_DIR = OUT_DIR / "runs_tbfcm_var_mix_lgf_go"  # 同一目录下按 variant 分子目
FIG_DIR = OUT_DIR / "figs_trueasm"
DATASET_DEFAULT = OUT_DIR / "dataset_trueasm_{variant}.npz"  # 动态

# 11 维特征: 9 featB + ret + roloc (与 featB_keys 顺序一致后再追加 ret/roloc)
FEAT_B_KEYS = ["eta2", "s3", "zeta2v", "so2v", "s2o2v",
               "zeta2e", "so2e", "s2o2e", "yw"]
EXTRA_FEATS = ["ret", "roloc"]
FEAT_NAMES = FEAT_B_KEYS + EXTRA_FEATS  # 11
SEL_BASIS_10 = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]  # 10基 (v/g/e 变体共用)
SEL_BASIS = SEL_BASIS_10  # 复用同名以减少 diff


# ============================================================================
# 1) 数据加载: pope_wy → 5 工况, 每工况 per-point 特征 + a_m + W_T_m + T_m 矩阵
# ============================================================================
def load_pope_wy(variant="v"):
    """返回 per-case 数据, 含 11 维特征 (valid) + a_m (10 列, NaN 标记无效) +
    每个基的 T 矩阵 ((N,6) sym6) + 每个基的 W 标量 ((N,) ‖T_m,ij‖²_F) +
    valid/valid_fit mask + b6 + yh + ro + k + eps.

    10 基反演: dataset_pope_wy 只存了 5 基 a_m (历史版本), 本函数**重新对
    10 基 T_m 做逐点 lstsq** 拿 10 列 a_m (与 cc10 口径一致)."""
    ds = np.load(DATASET)
    cases = []
    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        valid = ds[p + "valid"].astype(bool)
        # dataset_trueasm_* 只有 valid (127 点), 无 valid_fit; 用 valid 全集做 fit
        if (p + "valid_fit") in ds.files:
            vfit = ds[p + "valid_fit"].astype(bool)
        else:
            vfit = valid.copy()
            print(f"[load:{variant}] Ro={ro}: 无 valid_fit, 用 valid 全集 ({int(vfit.sum())} pts) 拟合")
        Nv = int(valid.sum())
        # 11 维特征: dataset_trueasm 用 c0_scalars (7) + yw, 而非 pope_wy 的 featB
        if (p + "featB") in ds.files:
            fb = ds[p + "featB"]                       # (N, 9)
        else:
            # dataset_trueasm_* 只有 c0_scalars (7): eta2,zeta2,s3,so2,s2o2,ret,roloc
            # 提取这 7 项并加 yw = (1-|yh|)*180
            scal = ds[p + "scalars"]                 # (N, 7)
            yh_local = ds[p + "yh"]
            yw = (1.0 - np.abs(yh_local)) * 180.0   # 真壁距
            # 取 zeta2=第1项(j=1, 因为 eta2=0), so2=第3(j=3), s2o2=第4(j=4) — 但 v/g/e 变体分别用 zeta2v/zeta2g/zeta2e 不同!
            # dataset_trueasm 的 scalars 是按 v/g/e 各自的涡量计算, 所以 zeta2 已经反映了 variant
            # 取: eta2=scal[:,0], zeta2=scal[:,1], s3=scal[:,2], so2=scal[:,3], s2o2=scal[:,4], yw=derived
            fb = np.column_stack([scal[:, 0], scal[:, 1], scal[:, 2],
                                  scal[:, 3], scal[:, 4], yw]).astype(float)  # (N, 6)
            # 然后 ret=ds[:,5], roloc=ds[:,6]
            # 拼成 11 维: [eta2, zeta2, s3, so2, s2o2, yw, (重复 yw, yw, yw 占位), ret, roloc]
            # 实际 dataset_pope_wy featB = [eta2, s3, zeta2v, so2v, s2o2v, zeta2e, so2e, s2o2e, yw]
            # (注意 s3 在前, 且 v/e 双套)
            # dataset_trueasm 只有一套; 凑 11 维: 用 [eta2, s3, zeta2, so2, s2o2, yw, yw, yw, yw, ret, roloc]
            fb = np.column_stack([
                scal[:, 0],          # eta2
                scal[:, 2],          # s3
                scal[:, 1],          # zeta2 (单通道, 对当前 variant)
                scal[:, 3],          # so2
                scal[:, 4],          # s2o2
                np.zeros_like(yw),   # zeta2e 占位 (变体无关)
                np.zeros_like(yw),   # so2e 占位
                np.zeros_like(yw),   # s2o2e 占位
                yw,                  # yw
            ]).astype(float)
        k = ds[p + "k"]; eps = ds[p + "eps"]; ro_f = float(ro)
        ret = 180.0 * k ** 2 / np.maximum(eps, 1e-20)  # Re_τ·k²/ε
        roloc = 2.0 * (ro_f / 360.0) * k / np.maximum(eps, 1e-20)
        feat = np.column_stack([fb, ret, roloc]).astype(float)  # (N, 11)
        # 每基 T 矩阵 (N, 6) 和 W 标量 (N,) = ‖T_m,ij‖²_F
        Tm = {m: ds[p + m].astype(float) for m in SEL_BASIS}  # dataset_trueasm: c{i}_T1..T10 (无 B_ 前缀)
        # W_Tm: 无 W key, 现算 ‖T_m,ij‖²_F
        Wm = {m: np.sum(Tm[m] ** 2, axis=1).astype(float) for m in SEL_BASIS}
        # === 10 基 lstsq 反演: 拿 10 列 a_m (与 cc10 口径一致) ===
        T_stack = np.stack([Tm[m] for m in SEL_BASIS], axis=2)  # (N, 6, 10)
        b6 = ds[p + "b6"].astype(float)                         # (N, 6)
        a10 = np.zeros((fb.shape[0], len(SEL_BASIS)), dtype=float)
        n_deg = 0
        for k_pt in range(fb.shape[0]):
            # 显式 SVD 截断到有效秩 (奇异值 > 1e-6 * 最大值): 求最小范数解
            U, S, Vt = np.linalg.svd(T_stack[k_pt], full_matrices=False)
            Smax = S.max() if len(S) else 0
            mask = S > max(1e-6 * Smax, 1e-12)
            r = int(mask.sum())
            if r == 0 or Smax < 0.1:
                # T 张量退化 (yh=±1 中心, 流动对称, S=W=0)
                a10[k_pt] = 0
                n_deg += 1
                # 同时把 vfit 标记为 False (后续 SR 不学这些点)
                if (p + "valid_fit") not in ds.files and vfit[k_pt]:
                    vfit[k_pt] = False
            else:
                # a = V @ S^{-1} @ U.T @ b (只在有效 r 上)
                a10[k_pt] = (Vt[:r].T @ ((U[:, :r].T @ b6[k_pt]) / S[:r]))
        if n_deg > 0 and i == 0:
            print(f"[load:lstsq10:{variant}] Ro={ro}: 剔除 {n_deg} 个 T 退化点 (Smax<0.1)")
        # === 与 np.load 互校: 5 基反演的 a (vs 10 基 lstsq) — 仅报告 ===
        if (p + "a") in ds.files:
            a5_old = ds[p + "a"].astype(float)
            if i == 0:
                print(f"[load:lstsq10:{variant}] Ro={ro}  a_max={np.max(np.abs(a10[vfit])):.2f}  "
                      f"a_max_5基={np.max(np.abs(a5_old[vfit])):.2f}")
        else:
            if i == 0:
                print(f"[load:lstsq10:{variant}] Ro={ro}  a_max={np.max(np.abs(a10[vfit])):.2f}  "
                      f"(无 a 旧值)")
        cases.append(dict(
            i=i, ro=ro_f, Nv=Nv,
            valid=valid, valid_fit=vfit,
            feat=feat, a=a10, T=Tm, W=Wm,
            b6=b6,   # (N, 6) DNS
            yh=ds[p + "yh"].astype(float),
            k=k, eps=eps))
        print(f"[load] Ro={ro:>5}  N={fb.shape[0]}  Nvalid={Nv}  Nvfit={int(vfit.sum())}")
    return cases


# ============================================================================
# 2) per-target SR 跑批 (minimax M2.7)
# ============================================================================
def build_target_cases(cases, m):
    """返回 LGF 训练需要的 cases: 5 工况, 每工况 {X, y, w, name}."""
    out = []
    for c in cases:
        vfit = c["valid_fit"]
        out.append(dict(
            X=c["feat"][vfit],          # (Nfit, 11)
            y=c["a"][vfit, SEL_BASIS.index(m)],  # (Nfit,)
            w=c["W"][m][vfit],          # (Nfit,) ‖T_m,ij‖²_F
            name=f"Ro{c['ro']:g}",
        ))
    return out


def physics_context_for(m):
    return f"""ASM (代数应力模型) 标量系数函数 g_m(I) 反演, 旋转槽道流 DNS,
Re_τ=180, Ro_τ=7.5/15/30/75/150 (Ω 沿流向 x1).
骨架: b_ij = Σ_m' g_m'(I)·T_m',ij, 其中 m' ∈ {{T1,T2v,T3v,T4v,T5v,T6v,T7v,T8v,T9v,T10v}}.
当前任务: 拟合 a_{m} = g_{m}(I) — 即张量基 T_{m} 的标量系数函数.

特征 (11 列, 标量不变量, 任何 y 截面上都对各分量共享):
  eta2, s3, zeta2v, so2v, s2o2v, zeta2e, so2e, s2o2e, yw, ret, roloc
  · yw = (1 - |yh|)·180: 真壁距, yw=0 在壁面, yw=180 中心
  · ret = Re_τ·k²/ε: 湍流尺度雷诺数 (跨 4 个量级, 近壁小、中心大)
  · roloc = 2·Ω⁺·k/ε: 局部旋转数 (中心值 5(Ro7.5)→127(Ro150))
  · 其它 8 项是应变率/涡量张量的无量纲不变量

物理提示:
  · a_{m} 是反演 DNS 得到的标量系数 (不是物理量), 跨 5 工况共享同一函数
  · 应力权重 w = ‖T_m,ij‖²_F, 强装配贡献区自带高梯度
  · 跨工况统一: 5 个 Ro 的 (X, y) 共同拟合, 公式中无 Ro 显式参数

【硬规则 — 违反公式直接判定为 reject, 不参与下一代】
  1. 禁 tanh, 禁 atanh — 大增益下硬饱和产生常数平台 (b12/b23 5 工况 60-90% 区域塌缩)
  2. 仅允许全局常数 C#; 禁逐工况 K# (违反 ASM 普适性, 真 ASM 硬约束)
  3. 增益护栏: |C·RMS(feature)| < 1.5 (避免 tanh-like 大增益结构, 不仅是 tanh 本身)
  4. 避免 exp(log(...)) 冗余结构
  5. 优选多项式、有理函数 (C0 + C1·x / (C2 + C3·x))、exp(-C/yw) 型壁面阻尼
  6. 推荐 smooth 形式, 跨 Ro 应平稳外推 (LGF-go 在低 Ro 拟合, 高 Ro 验证)
"""


def run_target_sr(m, cases, *, backend="minimax", gens=30, pop=50, budget=300,
                  sub_sample=300, max_edges=12, max_params=6, seed=0, variant="v"):
    """per-target LGF 跑批, 返回 (dsl, theta, loss, complexity, final_json_path)."""
    be = BACKENDS[backend]
    api_key = os.environ.get("LGF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(f"[tbfcm] LLM API key missing; abort.")
    client = llm_ops.make_client(be["backend"], url=be["url"], model=be["model"],
                                 api_key=api_key, timeout=240.0)
    print(f"[tbfcm-{m}] backend={backend} model={be['model']} cases="
          f"{[c['name'] for c in cases]} feats={len(FEAT_NAMES)}")
    s = (zlib.crc32(f"tbfcm|{m}|go".encode()) + seed) % (2 ** 31)
    cfg = LGFConfig(
        population=pop, generations=gens, sub_sample=sub_sample,
        time_budget_s=budget, max_edges=max_edges, max_params=max_params,
        final_starts=8, final_maxiter=150,
        llm_backend=be["backend"], llm_url=be["url"], llm_model=be["model"],
        llm_api_key=api_key, llm_timeout=240.0, seed=s,
        llm_ginit_only=True, wall_seeds=False,
        forbidden_ops=["tanh"],                # 硬规则 1+3 (硬禁 tanh)
        physics_context=physics_context_for(m).replace("{variant}", variant),
        global_params_only=True,                # 硬规则 2
        ginit_retry=2, ginit_min_kept=3, ginit_fewshot=True,
    )
    out_dir = RUN_DIR / variant / m
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_lgf(cases, FEAT_NAMES, cfg, out_dir, client, name=m)
    elapsed = time.time() - t0
    best = res["best"]
    # 禁 tanh 后置检查 (兜底)
    if "tanh" in best["dsl"] or "atanh" in best["dsl"]:
        print(f"[tbfcm-{m}] ⚠ 违反禁 tanh 硬规则, 仍返回; 装配时按规则 6 拒绝")
    print(f"[tbfcm-{m}] best loss={best['loss']:.5f}  cplx={best['complexity']}  "
          f"elapsed={elapsed:.0f}s  ::  {best['dsl'][:120]}")
    return best, out_dir, elapsed


# ============================================================================
# 3) 装配 + 评估
# ============================================================================
def compile_global(m, dsl):
    """编译单基的标量公式, 返回 CompiledGraph + case_names 检查."""
    g = lgfdsl.parse_dsl(dsl, FEAT_NAMES)
    if g.case_names():
        raise ValueError(f"[{m}] DSL 含逐工况常数 K: {g.case_names()}")
    return CompiledGraph(g, backend="numpy")


def predict_a(m, cg, theta, X):
    """对单基 m, 给定 11 维特征 X, 预测 a_m (N,)."""
    return np.asarray(cg.eval_case(np.asarray(theta, dtype=float), X), dtype=float)


def assemble(cases, formulas, *, eval_mask="valid"):
    """装配 b_pred = Σ_m â_m(I_pt)·T_m,ij, 返回 per-case (Nv_eval, 6) 矩阵列表.

    formulas: dict m -> (dsl, theta)
    eval_mask: "valid" (126) 或 "valid_fit" (122, 训练集内)
    """
    compiled = {m: compile_global(m, dsl) for m, (dsl, _) in formulas.items()}
    out = []
    for c in cases:
        msk = c[eval_mask]
        X = c["feat"][msk]                   # (Ne, 11)
        Ne = X.shape[0]
        b_pred = np.zeros((Ne, 6), dtype=float)
        for m in SEL_BASIS:
            a_pred = predict_a(m, compiled[m], formulas[m][1], X)  # (Ne,)
            b_pred += a_pred[:, None] * c["T"][m][msk]              # (Ne, 6)
        out.append(b_pred)
    return out


def build_eval_data(cases, *, eval_mask="valid"):
    """构造 rc_asm_true_eval 风格的 data, 每工况 flat b6 (6*Nv,) + 必要元数据."""
    data = []
    for c in cases:
        msk = c[eval_mask]
        b6v = c["b6"][msk]                    # (Ne, 6)
        yh = c["yh"][msk]                     # (Ne,)
        b6_flat = b6v.reshape(-1)             # (6*Ne,)
        # 构造"假 Xfit/yfit" (eval_predictions 不真用, 只读 size);
        # var_c/max_abs_ref 反映 b6 展平后的尺度
        data.append(dict(
            Nv=b6v.shape[0], b6v=b6v, yh=yh, ro=c["ro"], i=c["i"],
            Xfit=np.zeros((6 * b6v.shape[0], 1), dtype=float),  # dummy
            yfit=b6_flat,                                       # for var_c/max_abs_ref
            var_c=float(np.var(b6_flat)),
            max_abs_ref=float(np.max(np.abs(b6_flat))),
        ))
    return data


def evaluate(cases, formulas, *, eval_mask="valid", label="tbfcm"):
    """装配 + 调用 eval_predictions, 返回统一度量 dict."""
    asm_preds = assemble(cases, formulas, eval_mask=eval_mask)
    data = build_eval_data(cases, eval_mask=eval_mask)
    # eval_predictions 期望 preds 是 list of 1-D (6*Nv,) flat 数组
    preds_flat = [p.reshape(-1) for p in asm_preds]
    m = eval_predictions(preds_flat, data)
    # 附加: 每分量 (6 个) 的 NMSE 表
    per_comp = {}
    for ic, comp in enumerate(COMP6):
        comp_nmse = []
        for ci, c in enumerate(cases):
            msk = c[eval_mask]
            truth = c["b6"][msk, ic]
            pred = asm_preds[ci][:, ic]
            var_t = float(np.var(truth)) + 1e-12
            mse = float(np.mean((pred - truth) ** 2))
            comp_nmse.append(mse / var_t)
        per_comp[comp] = comp_nmse
    m["per_comp_nmse"] = per_comp
    m["per_comp_mean"] = {k: float(np.mean(v)) for k, v in per_comp.items()}
    m["label"] = label
    m["eval_mask"] = eval_mask
    return m, asm_preds


# ============================================================================
# 4) 基线: cc10 (10 个全局常数 lstsq), oracle (用反演的 a_m), mix_lgf_go, mix_pysr
# ============================================================================
def baseline_cc10(cases, *, eval_mask="valid"):
    """c·T 全局常数联合 lstsq, 与 TBFCM 同装配路径, 作为 TBFCM 性能上限参考.
    注: 与 mix report 中的 cc10 不同, 本处是 v 通道的 10 个基, 用于直接对比."""
    full_basis = SEL_BASIS_10  # 10基 (当前 variant)
    # cc10 = 10 个全局常数 c_m, 跨工况共同
    # 拼 X: (sum_e 6*Ne, 10) 与 y: (sum_e 6*Ne,)
    Xs, ys = [], []
    for c in cases:
        msk = c[eval_mask]
        Ne = msk.sum()
        for m in full_basis:
            if m not in c["T"]:
                # 缺基 (dataset_pope_wy 只存了 5 个) → 用 0 列
                Xs.append(np.zeros(6 * Ne)); continue
            T_full = np.tile(c["T"][m][msk].reshape(-1)[:, None], (1, 1))  # (6*Ne, 1)
            Xs.append(T_full.reshape(-1))
        # 实际简化: 对每个基, T 矩阵是 (Ne, 6) sym6 → flat (6*Ne,)
        # 重做: 对每个 m 拆列
    # 重写更简单版本
    Xs = []; ys = []
    for c in cases:
        msk = c[eval_mask]
        Ne = int(msk.sum())
        b6_flat = c["b6"][msk].reshape(-1)   # (6*Ne,)
        # X 矩阵: 每行对应 b6_flat 一项, 列 = 每个基的 T_{m,ij} 对应分量
        Xm = np.zeros((6 * Ne, len(full_basis)), dtype=float)
        for j, m in enumerate(full_basis):
            if m in c["T"]:
                Xm[:, j] = c["T"][m][msk].reshape(-1)
        Xs.append(Xm); ys.append(b6_flat)
    X = np.vstack(Xs); y = np.concatenate(ys)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    # 装配预测
    asm_preds = []
    offset = 0
    for c in cases:
        msk = c[eval_mask]; Ne = int(msk.sum())
        Xm = np.zeros((6 * Ne, len(full_basis)), dtype=float)
        for j, m in enumerate(full_basis):
            if m in c["T"]:
                Xm[:, j] = c["T"][m][msk].reshape(-1)
        pred_flat = Xm @ coef
        asm_preds.append(pred_flat.reshape(Ne, 6))
    data = build_eval_data(cases, eval_mask=eval_mask)
    preds_flat = [p.reshape(-1) for p in asm_preds]
    metrics = eval_predictions(preds_flat, data)
    # per-comp
    per_comp = {}
    for ic, comp in enumerate(COMP6):
        per_comp[comp] = []
        for ci, c in enumerate(cases):
            msk = c[eval_mask]
            truth = c["b6"][msk, ic]
            pred = asm_preds[ci][:, ic]
            var_t = float(np.var(truth)) + 1e-12
            per_comp[comp].append(float(np.mean((pred - truth) ** 2) / var_t))
    metrics["per_comp_nmse"] = per_comp
    metrics["per_comp_mean"] = {k: float(np.mean(v)) for k, v in per_comp.items()}
    metrics["label"] = "cc10_lstsq"
    metrics["eval_mask"] = eval_mask
    metrics["coef"] = coef.tolist()
    return metrics, asm_preds, coef


def baseline_oracle(cases, *, eval_mask="valid"):
    """用反演 a_m 精确装配 (b_true = Σ a_m·T_m,ij), 给"结构表达力"上限."""
    asm_preds = []
    for c in cases:
        msk = c[eval_mask]
        Ne = int(msk.sum())
        b_pred = np.zeros((Ne, 6), dtype=float)
        a_m = c["a"][msk]  # (Ne, 5)
        for j, m in enumerate(SEL_BASIS):
            b_pred += a_m[:, j:j+1] * c["T"][m][msk]
        asm_preds.append(b_pred)
    data = build_eval_data(cases, eval_mask=eval_mask)
    preds_flat = [p.reshape(-1) for p in asm_preds]
    metrics = eval_predictions(preds_flat, data)
    per_comp = {}
    for ic, comp in enumerate(COMP6):
        per_comp[comp] = []
        for ci, c in enumerate(cases):
            msk = c[eval_mask]
            truth = c["b6"][msk, ic]
            pred = asm_preds[ci][:, ic]
            var_t = float(np.var(truth)) + 1e-12
            per_comp[comp].append(float(np.mean((pred - truth) ** 2) / var_t))
    metrics["per_comp_nmse"] = per_comp
    metrics["per_comp_mean"] = {k: float(np.mean(v)) for k, v in per_comp.items()}
    metrics["label"] = "oracle_sel5"
    metrics["eval_mask"] = eval_mask
    return metrics, asm_preds


def load_existing(label, run_subdir, kind="lgf"):
    """从已有 run 加载评估结果 (loss_asm/rho_a/per_comp), 复用为对比基线.

    kind: "lgf" (trueasm mix_lgf_go) 或 "pysr" (trueasm mix_pysr)
    """
    p = OUT_DIR / run_subdir / "result.json"
    if not p.exists():
        return None
    j = json.load(open(p, encoding="utf-8"))
    if kind == "lgf" and "entries" in j:
        b = j["entries"][0]; m = b.get("metrics", {}) or {}
        dsl = b.get("dsl", "")
    elif kind == "pysr" and "best" in j:
        b = j["best"]; m = b.get("metrics", {}) or {}
        dsl = b.get("sympy", "")
    else:
        return None
    return dict(label=label, loss_asm=m.get("loss_asm"), rho_a=m.get("rho_a"),
                per_case=m.get("per_case", {}), dsl=dsl, source=str(p))


# ============================================================================
# 5) 平坦区 (常数区域) 比例 — 验收硬规则 6
# ============================================================================
def plateau_fraction(asm_preds, cases, eval_mask="valid", comp_idx=None,
                     thresh=0.1):
    """对每个 (case, comp), 计算预测剖面中 |d pred/dy·h| < thresh·std 的点占比.

    简化: 沿 yh 排序后用 |Δpred| / Δyh < thresh·std(pred) 的判据.
    """
    rows = []
    for ci, c in enumerate(cases):
        msk = c[eval_mask]; yh = c["yh"][msk]; b_pred = asm_preds[ci]
        Ne = yh.shape[0]
        order = np.argsort(yh)
        for ic, comp in enumerate(COMP6):
            if comp_idx is not None and ic != comp_idx:
                continue
            p = b_pred[order, ic]
            std = float(np.std(p)) + 1e-12
            # 局部斜率 (相邻点差分)
            dyp = np.diff(p)
            dyh = np.diff(yh[order])
            slope = np.abs(dyp / np.where(np.abs(dyh) > 1e-12, dyh, 1e-12))
            flat_frac = float(np.mean(slope < thresh * std))
            rows.append((c["ro"], comp, flat_frac))
    return rows


# ============================================================================
# 6) 主流程
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=SEL_BASIS,
                    help="要拟合的基, 默认全 10 个")
    ap.add_argument("--variant", default="v", choices=["v", "g", "e"],
                    help="变体 (v=涡量, g=Gatski-Speziale, e=扩展内瞒)")
    ap.add_argument("--backend", default="minimax", choices=list(BACKENDS))
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--sub_sample", type=int, default=300)
    ap.add_argument("--max_edges", type=int, default=12)
    ap.add_argument("--max_params", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_sr", action="store_true",
                    help="跳过 SR, 复用 runs_tbfcm_var_mix_lgf_go/{variant}/<m>_final.json")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_report", action="store_true")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # 动态选择数据集
    global DATASET
    DATASET = OUT_DIR / f"dataset_trueasm_{args.variant}.npz"
    cases = load_pope_wy(variant=args.variant)

    # ---- Stage 1: 5 目标 SR ----
    formulas = {}  # m -> (dsl, theta)
    sr_meta = {}   # m -> 跑批元数据
    if not args.skip_sr:
        for m in args.targets:
            assert m in SEL_BASIS, f"未知基 {m}, 必须在 {SEL_BASIS}"
            print(f"\n========== TBFCM-SR target={m} ==========")
            tgt_cases = build_target_cases(cases, m)
            best, out_dir, elapsed = run_target_sr(
                m, tgt_cases, backend=args.backend, gens=args.gens,
                pop=args.pop, budget=args.budget, sub_sample=args.sub_sample,
                max_edges=args.max_edges, max_params=args.max_params,
                seed=args.seed, variant=args.variant)
            formulas[m] = (best["dsl"], best["theta"])
            sr_meta[m] = dict(loss=best["loss"], complexity=best["complexity"],
                              dsl=best["dsl"], theta=best["theta"],
                              elapsed_s=round(elapsed, 1))
            # 立即落盘 (避免后续 SR 失败丢失)
            mid = dict(target=m, **sr_meta[m],
                       feat_names=FEAT_NAMES, sel_basis=SEL_BASIS,
                       seed=args.seed, backend=args.backend)
            with open(out_dir / "tbfcm_var_target_meta.json", "w", encoding="utf-8") as f:
                json.dump(mid, f, ensure_ascii=False, indent=2, default=str)
    else:
        for m in args.targets:
            j = json.load(open(RUN_DIR / args.variant / m / "tbfcm_var_target_meta.json", encoding="utf-8"))
            formulas[m] = (j["dsl"], j["theta"])
            sr_meta[m] = j
        print(f"[tbfcm] skip_sr: 复用 {len(formulas)} 个目标 {list(formulas)}")

    if args.skip_eval:
        print("[tbfcm] skip_eval: 评估/报告跳过")
        return

    # ---- Stage 2: 装配 + 评估 ----
    # 只装配有公式的目标
    fit_targets = [m for m in SEL_BASIS if m in formulas]
    print(f"\n========== TBFCM 装配: {fit_targets} ==========")
    tbfcm_metrics, tbfcm_preds = evaluate(cases, formulas, eval_mask="valid", label="tbfcm")
    # 训练集内 (valid_fit) 评估 — 看是否过拟合
    tbfcm_metrics_fit, _ = evaluate(cases, formulas, eval_mask="valid_fit",
                                    label="tbfcm_fit")
    print(f"[tbfcm] valid     loss_asm={tbfcm_metrics['loss_asm']:.4f}  "
          f"rho_a={tbfcm_metrics['rho_a']:+.4f}")
    print(f"[tbfcm] valid_fit loss_asm={tbfcm_metrics_fit['loss_asm']:.4f}  "
          f"rho_a={tbfcm_metrics_fit['rho_a']:+.4f}")

    # ---- Stage 3: 基线对比 ----
    cc10_metrics, cc10_preds, cc10_coef = baseline_cc10(cases, eval_mask="valid")
    oracle_metrics, oracle_preds = baseline_oracle(cases, eval_mask="valid")
    mix_lgf_go = load_existing("mix_lgf_go", "runs_trueasm_mix_lgf_go", "lgf")
    mix_pysr = load_existing("mix_pysr", "runs_trueasm_mix_pysr", "pysr")
    print(f"[cc10] loss_asm={cc10_metrics['loss_asm']:.4f}  "
          f"rho_a={cc10_metrics['rho_a']:+.4f}  coef={cc10_coef}")
    print(f"[oracle_sel5] loss_asm={oracle_metrics['loss_asm']:.4f}  "
          f"rho_a={oracle_metrics['rho_a']:+.4f}")

    # 平坦区比例 (硬规则 6)
    plateau_tbfcm = plateau_fraction(tbfcm_preds, cases, eval_mask="valid",
                                     thresh=0.1)
    plateau_cc10 = plateau_fraction(cc10_preds, cases, eval_mask="valid", thresh=0.1)
    # mix_lgf_go 剖面 (从 preds_best.npz 读)
    plateau_mix_lgf_go = None
    plgf = OUT_DIR / "runs_trueasm_mix_lgf_go" / "preds_best.npz"
    if plgf.exists():
        ds = np.load(plgf)
        # mix_lgf_go preds 是 (6*Nv,) flat, 需 reshape 到 (Nv, 6)
        mix_preds = []
        for c in cases:
            msk = c["valid"]; Ne = int(msk.sum())
            # 注: trueasm mix 的 valid 集是 127, pope_wy valid 是 126
            # 对每工况: 该数据集的"valid"数 = ds[c{i}].shape[0] // 6
            raw = ds[f"c{c['i']}"] if f"c{c['i']}" in ds.files else None
            if raw is None or raw.size != 6 * Ne:
                # 长度对不上, 跳过此工况
                mix_preds.append(None); continue
            mix_preds.append(raw.reshape(Ne, 6))
        # 简单粗暴: 只对长度匹配工况算 plateau
        rows = []
        for ci, c in enumerate(cases):
            if mix_preds[ci] is None: continue
            msk = c["valid"]; yh = c["yh"][msk]; b_pred = mix_preds[ci]
            order = np.argsort(yh)
            for ic, comp in enumerate(COMP6):
                p = b_pred[order, ic]
                std = float(np.std(p)) + 1e-12
                slope = np.abs(np.diff(p) / np.diff(yh[order]))
                rows.append((c["ro"], comp, float(np.mean(slope < 0.1 * std))))
        plateau_mix_lgf_go = rows

    # ---- Stage 4: 汇总落盘 ----
    summary = dict(
        label=f"tbfcm_var_{variant}_lgf_go",
        method="lgf-go (per-target SR, 5 targets)",
        backend=args.backend,
        model=BACKENDS[args.backend]["model"],
        feat_names=FEAT_NAMES, sel_basis=SEL_BASIS,
        targets_attempted=list(args.targets),
        targets_fitted=fit_targets,
        config=dict(gens=args.gens, pop=args.pop, budget=args.budget,
                    sub_sample=args.sub_sample, max_edges=args.max_edges,
                    max_params=args.max_params, seed=args.seed,
                    llm_ginit_only=True, global_params_only=True,
                    forbidden_ops=["tanh"]),
        per_target=sr_meta,
        metrics_valid=tbfcm_metrics,
        metrics_valid_fit=tbfcm_metrics_fit,
        baseline_cc10=cc10_metrics,
        baseline_oracle_sel5=oracle_metrics,
        baseline_mix_lgf_go=mix_lgf_go,
        baseline_mix_pysr=mix_pysr,
        plateau=dict(tbfcm=plateau_tbfcm, cc10=plateau_cc10,
                     mix_lgf_go=plateau_mix_lgf_go),
        cc10_coef=cc10_coef.tolist(),
    )
    out_json = RUN_DIR / variant / "tbfcm_var_result.json"  # 按 variant 拆分
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[tbfcm] -> {out_json}")

    if args.skip_report:
        return

    # ---- Stage 5: 报告 ----
    write_report(summary, out_json.with_suffix(".md"), variant=args.variant)


def write_report(s, md_path, variant="v"):
    m = s["metrics_valid"]
    mfit = s["metrics_valid_fit"]
    cc10 = s["baseline_cc10"]; oracle = s["baseline_oracle_sel5"]
    mix_lgf_go = s["baseline_mix_lgf_go"]; mix_pysr = s["baseline_mix_pysr"]
    # 表格行
    rows = []
    rows.append(("TBFCM (本)", f"{m['loss_asm']:.4f}", f"{m['rho_a']:+.4f}",
                 f"{m['per_comp_mean'].get('uv', float('nan')):.3f}",
                 f"{m['per_comp_mean'].get('vw', float('nan')):.3f}"))
    rows.append(("cc10 (10常数 lstsq)", f"{cc10['loss_asm']:.4f}", f"{cc10['rho_a']:+.4f}",
                 f"{cc10['per_comp_mean'].get('uv', float('nan')):.3f}",
                 f"{cc10['per_comp_mean'].get('vw', float('nan')):.3f}"))
    rows.append(("oracle_sel5 (反演 a_m)",
                 f"{oracle['loss_asm']:.4f}", f"{oracle['rho_a']:+.4f}",
                 f"{oracle['per_comp_mean'].get('uv', float('nan')):.3f}",
                 f"{oracle['per_comp_mean'].get('vw', float('nan')):.3f}"))
    if mix_lgf_go:
        rows.append(("mix_lgf_go (单公式 tanh)",
                     f"{mix_lgf_go['loss_asm']:.4f}", f"{mix_lgf_go['rho_a']:+.4f}",
                     "—", "—"))
    if mix_pysr:
        rows.append(("mix_pysr (PySR 全特征)",
                     f"{mix_pysr['loss_asm']:.4f}", f"{mix_pysr['rho_a']:+.4f}",
                     "—", "—"))

    # per-comp NMSE 表 (TBFCM vs cc10)
    pcomp_rows = []
    for comp in COMP6:
        tbfcm_c = m["per_comp_mean"].get(comp, float("nan"))
        cc10_c = cc10["per_comp_mean"].get(comp, float("nan"))
        oracle_c = oracle["per_comp_mean"].get(comp, float("nan"))
        pcomp_rows.append((comp, tbfcm_c, cc10_c, oracle_c))

    # 5 个公式表
    target_rows = []
    for m_name in s["sel_basis"]:
        if m_name in s["per_target"]:
            t = s["per_target"][m_name]
            target_rows.append((m_name, t["loss"], t["complexity"],
                                t["dsl"][:80] + ("…" if len(t["dsl"]) > 80 else ""),
                                t.get("elapsed_s", 0)))

    md = f"""# TBFCM 变体 10 基混合训练报告 (variant={s.get("variant", variant)}, 2026-10-10)

**骨架**: b_ij(I) = Σ_m g_m(I)·T_m,ij, g_m 由 per-target LGF-go 独立反演.

**基**: {', '.join(s['sel_basis'])} (沿用 dataset_pope_wy)

**特征** (11 列): {', '.join(s['feat_names'])}

**护栏**: forbidden_ops=["tanh"], global_params_only=True, llm_ginit_only=True

**后端**: {s['backend']} ({s['model']}), gens={s['config']['gens']},
pop={s['config']['pop']}, budget={s['config']['budget']}s/target, seed={s['config']['seed']}

## 1️⃣ 装配结果 (eval on valid, N=126/case)

| 方法 | loss_asm | ρ_a | b12 NMSE | b23 NMSE |
|---|---|---|---|---|
""" + "\n".join(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |"
        for r in rows) + f"""

**训练集内 (valid_fit, N=122/case)**:
TBFCM loss_asm = {mfit['loss_asm']:.4f}, ρ_a = {mfit['rho_a']:+.4f}

## 2️⃣ 逐分量 NMSE (5 工况平均, valid)

| 分量 | TBFCM | cc10 | oracle_sel5 |
|---|---|---|---|
""" + "\n".join(
        f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} |" for r in pcomp_rows) + f"""

## 3️⃣ 5 个 g_m 公式

| 基 | SR loss | cplx | DSL | 用时(s) |
|---|---|---|---|---|
""" + "\n".join(
        f"| {r[0]} | {r[1]:.4f} | {r[2]} | `{r[3]}` | {r[4]:.0f} |"
        for r in target_rows) + f"""

## 4️⃣ 平坦区比例 (硬规则 6 检验)

方法判据: |dpred/dy|·h < 0.1·std(pred) 的点占比.
""" + plateau_table(s["plateau"]) + f"""

**接受阈值 (待定)**: 若 TBFCM 平坦区比例 ≪ mix_lgf_go 同分量 → 方案 1 生效.

## 5️⃣ 结论 (待填: 跑完装配后回写)

---
产物:
- `outputs/asm/runs_tbfcm10_mix_lgf_go/tbfcm10_result.json` (完整度量)
- `outputs/asm/runs_tbfcm10_mix_lgf_go/{{m}}/tbfcm10_target_meta.json` (per-target)
- `outputs/asm/ASM_tbfcm10_mix_report.md` (本报告)
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"[tbfcm] -> {md_path}")


def plateau_table(plat):
    # plat = {tbfcm: [(ro, comp, frac), ...], cc10: ..., mix_lgf_go: ...}
    methods = [k for k in ("tbfcm", "cc10", "mix_lgf_go") if plat.get(k)]
    if not methods: return "_无数据_\n"
    # 汇总 (mean over 5 Ro) per method per comp
    comps = ["uu", "uv", "uw", "vv", "vw", "ww"]
    out = ["| comp | " + " | ".join(methods) + " |", "|---|---|" + "|".join(["---"] * len(methods)) + "|"]
    for comp in comps:
        row = [comp]
        for k in methods:
            vals = [f for (ro, c, f) in plat[k] if c == comp]
            row.append(f"{np.mean(vals):.3f}" if vals else "—")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
