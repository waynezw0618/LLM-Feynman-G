# -*- coding: utf-8 -*-
"""rc_asm_basisB2.py — 方案B v2: **正交化(白化)张量基**口径的 b23 闭合.

问题 (2026-10-16 诊断): 直接用原始 T̃ 通道做 "b23 = Σ_m a_m(Θ)·T̃_m23",
其系数 a_m 幅值巨大 (10^2~10^3), 各通道项需相互抵消 10^2~10^4 倍才得到净
b23 (Ro150 抵消比 2.5e4). 后果: 系数自身拟合到 1e-6, 重组 b23 却爆到 3.8~4.5.

修正 (v2): 对张量通道做**加权白化**.
  G = Σ_i w_i x_i x_i^T ,   w_i = 1/(Nv·var_c(b23))  (逐工况 NMSE 权重)
  G = V Λ V^T,   W = V Λ^{-1/2}   (截断 λ ≤ tol·λ_max 的方向)
  U_i = W^T x_i   (白化通道, Σ_i w_i U_i U_i^T = I)
于是 b23 = Σ_j β_j(Θ)·U_j 中的 β_j 满足 |β|² ≤ Σ_i w_i b23_i² = 5,
即 **O(1) 且唯一(最小范数)**, 病态抵消消失; 且因 U 加权正交, 逐通道独立拟合
与联合拟合等价. 最终仍可还原回原始 T 基: a = W β → b23 = Σ_m a_m·T̃_m.

两阶段:
  stage base — 结构化联合加权 lstsq 定系数展开 a_j(Θ)=Σ_k c_jk φ_k(Θ), 给出
               结构化基线精度 + 每条 β_j 的逐点靶标;
  stage sr   — LGF(lgf_go/lgf_mg)/PySR 对每条 β_j(Θ) 做符号回归, 再重组评估.

评估口径 (同 rc_asm_vw_only.eval_vw_predictions):
  loss_vw = mean_c[ mean_y (b23_pred-b23)² / var_c(b23) ].

用法:
  python -X utf8 scripts/rc_asm_basisB2.py --stage base --variant g
  python -X utf8 scripts/rc_asm_basisB2.py --stage sr --variant g --method pysr
  python -X utf8 scripts/rc_asm_basisB2.py --stage sr --variant g --method lgf_go --backend glm53
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rc_asm_run import BACKENDS  # noqa: E402
from lgf.config import LGFConfig  # noqa: E402
from lgf import llm_ops  # noqa: E402
from lgf.pipeline_lgf import run_lgf  # noqa: E402
from lgf.compiler import CompiledGraph  # noqa: E402
from lgf import dsl as lgfdsl  # noqa: E402

OUT = REPO / "outputs" / "asm"
ROS = [7.5, 15, 30, 75, 150]
IDX23 = 5
CHANS = [1, 5, 6, 10]
BASE = ["eta2", "zeta2", "s2o2", "ret", "roloc", "yw"]

# Θ 系数基池 (bounded pool): 全部基 |φ|<=O(10), 保证系数 a_m 良态 O(1)
POOL = [
    ("1", lambda f: np.ones_like(f["yw"])),
    ("sqrt(rl)/(1+sqrt(rl))", lambda f: np.sqrt(f["roloc"]) / (1 + np.sqrt(f["roloc"]))),
    ("tanh(roloc)", lambda f: np.tanh(f["roloc"])),
    ("roloc/(1+roloc)", lambda f: f["roloc"] / (1 + f["roloc"])),
    ("eta2/(1+eta2)", lambda f: f["eta2"] / (1 + f["eta2"])),
    ("zeta2/(1+zeta2)", lambda f: f["zeta2"] / (1 + f["zeta2"])),
    ("s2o2/(1e4+|s2o2|)", lambda f: f["s2o2"] / (1e4 + np.abs(f["s2o2"]))),
    ("log((eta2+.01)/(eta2+1))", lambda f: np.log((f["eta2"] + 1e-2) / (f["eta2"] + 1))),
    ("log1p(ret)", lambda f: np.log1p(f["ret"])),
    ("yw", lambda f: f["yw"]),
    ("yw^2", lambda f: f["yw"] ** 2),
    ("yw*sqrt(rl)/(1+sqrt(rl))", lambda f: f["yw"] * np.sqrt(f["roloc"]) / (1 + np.sqrt(f["roloc"]))),
]

# 旧的无界池 (历史对照)
POOL_OLD = [
    ("1", lambda f: np.ones_like(f["yw"])),
    ("sqrt(roloc)", lambda f: np.sqrt(f["roloc"])),
    ("roloc", lambda f: f["roloc"]),
    ("tanh(roloc)", lambda f: np.tanh(f["roloc"])),
    ("eta2", lambda f: f["eta2"]),
    ("zeta2", lambda f: f["zeta2"]),
    ("s2o2", lambda f: f["s2o2"]),
    ("ret", lambda f: f["ret"]),
    ("log(eta2)", lambda f: np.log(f["eta2"] + 1e-6)),
    ("yw", lambda f: f["yw"]),
    ("yw^2", lambda f: f["yw"] ** 2),
    ("yw*sqrt(roloc)", lambda f: f["yw"] * np.sqrt(f["roloc"])),
]

# 紧凑有界池 (coefficient-pool B): 所有基 O(1)-O(10), 避免系数爆炸
POOL_C = [
    ("1", lambda f: np.ones_like(f["yw"])),
    ("sqrt(roloc)", lambda f: np.sqrt(f["roloc"])),
    ("tanh(roloc)", lambda f: np.tanh(f["roloc"])),
    ("eta2/(1+eta2)", lambda f: f["eta2"] / (1.0 + f["eta2"])),
    ("zeta2/(1+zeta2)", lambda f: f["zeta2"] / (1.0 + f["zeta2"])),
    ("log1p(ret)", lambda f: np.log1p(f["ret"])),
    ("log(eta2)", lambda f: np.log(f["eta2"] + 1e-6)),
    ("yw", lambda f: f["yw"]),
    ("yw^2", lambda f: f["yw"] ** 2),
]
POOLS = {"rich": POOL, "compact": POOL_C, "unbounded": POOL_OLD}

# T23 通道名 (用于报告)
M_POOL_CHAN = ["T1s"] + [f"{t}s_{v}" for t in ("T5", "T6", "T10") for v in ("v", "g", "e")]
NAMED = {"v": ["T1s", "T5s", "T6s", "T10s"], "g": ["T1s", "T5s", "T6s", "T10s"],
         "e": ["T1s", "T5s", "T6s", "T10s"], "m": M_POOL_CHAN}


# ---------------------------------------------------------------------------
def load_cases(variant: str, cnorm: str = "pooled"):
    ds = None
    dss = None
    if variant == "m":
        dss = {v: np.load(OUT / f"dataset_trueasm_{v}.npz") for v in ("v", "g", "e")}
    else:
        ds = np.load(OUT / f"dataset_trueasm_{variant}.npz")
    cases = []
    for i, ro in enumerate(ROS):
        p = f"c{i}_"
        if variant == "m":
            src = dss["v"]
            valid = src[p + "valid"].astype(bool)
            yh = src[p + "yh"][valid].astype(float)
            sc = src[p + "scalars"][valid].astype(float)
            b23 = src[p + "b6"][valid, IDX23].astype(float)
            cols = []
            for t in ("T1", "T5", "T6", "T10"):
                srcs = ("v",) if t == "T1" else ("v", "g", "e")
                for v in srcs:
                    cols.append(dss[v][p + t][valid, IDX23].astype(float))
            ch = np.column_stack(cols)
        else:
            valid = ds[p + "valid"].astype(bool)
            yh = ds[p + "yh"][valid].astype(float)
            sc = ds[p + "scalars"][valid].astype(float)
            b23 = ds[p + "b6"][valid, IDX23].astype(float)
            ch = np.column_stack([ds[p + f"T{m}"][valid, IDX23].astype(float)
                                  for m in CHANS])
        f = dict(eta2=sc[:, 0], zeta2=sc[:, 1], s2o2=sc[:, 4], ret=sc[:, 5],
                 roloc=sc[:, 6], yw=1.0 - np.abs(yh))
        if cnorm == "percase":
            rr = np.sqrt(np.mean(ch ** 2, 0))
            rr[rr <= 0] = 1.0
            ch = ch / rr
        cases.append(dict(ro=ro, yh=yh, b23=b23, f=f, ch=ch, Nv=len(yh),
                          var_c=float(np.var(b23))))
    return cases


def weight(cases):
    return np.concatenate([np.full(c["Nv"], 1.0 / (c["Nv"] * c["var_c"]))
                           for c in cases])


def _sfx(train_ros):
    """留一/子集拆分时的文件后缀 (全工况时为 '')."""
    if train_ros and set(train_ros) != set(ROS):
        return "_tr" + "-".join(f"{r:g}" for r in train_ros)
    return ""


def _tpaths(variant, train_ros):
    s = _sfx(train_ros)
    return OUT / f"basisB2_targets_{variant}{s}.npz", OUT / f"basisB2_base_{variant}{s}.json"


def whiten(cases, jmax, tol, use_whiten=True):
    """加权白化. 返回 W (Jch×J), lam(降序), J, 白化后各 case 的 U."""
    w = weight(cases)
    X = np.vstack([c["ch"] for c in cases])
    G = X.T @ (X * w[:, None])
    if not use_whiten:
        lam = np.diag(G).copy()
        W = np.eye(X.shape[1])
        for c in cases:
            c["U"] = c["ch"]
        return W, lam, X.shape[1]
    lam, V = np.linalg.eigh(G)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    lam[lam < 0] = 0.0
    keep = lam > tol * lam[0]
    J = int(min(int(keep.sum()), jmax))
    lam_k, V_k = lam[:J], V[:, :J]
    W = V_k / np.sqrt(lam_k)[None, :]
    for c in cases:
        c["U"] = c["ch"] @ W
    return W, lam, J


def wlstsq(Z, y, w, ridge=0.0):
    """加权最小二乘 (列归一化). ridge>0 时为岭回归, 取最小范数解.

    病态性根源: 张量通道近共线 → 无正则解有巨大且相消的系数 (β_rms~1e5),
    使系数拟合的微小误差被放大到 b23 量级之上. 岭正则把它压回 O(1).
    """
    sw = np.sqrt(w)
    Zs, ys = Z * sw[:, None], y * sw
    sc = np.sqrt((Zs ** 2).sum(0))
    sc[sc <= 0] = 1.0
    Zn = Zs / sc
    if ridge > 0:
        A = Zn.T @ Zn
        A[np.diag_indices_from(A)] += ridge
        cs = np.linalg.solve(A, Zn.T @ ys)
    else:
        cs, *_ = np.linalg.lstsq(Zn, ys, rcond=1e-11)
    return cs / sc


def eval_nmse(pred_list, cases):
    pc, nms = {}, []
    for p, c in zip(pred_list, cases):
        r = float(np.mean((np.asarray(p) - c["b23"]) ** 2) / c["var_c"])
        nms.append(r)
        pc[f"Ro{c['ro']:g}"] = round(r, 4)
    return dict(loss_vw=float(np.mean(nms)), worst=float(np.max(nms)), per_case=pc)


# ---------------------------------------------------------------------------
def _design_U(cases, pool):
    """列 = φ_k(Θ)·U_j."""
    K = len(pool)
    Z, names = [], []
    for c in cases:
        B = np.column_stack([fn(c["f"]) for _, fn in pool])          # Nv×K
        J = c["U"].shape[1]
        blk = (B[:, :, None] * c["U"][:, None, :]).reshape(c["Nv"], K * J)
        Z.append(blk)
    names = [f"{kn}·U{j+1}" for kn, _ in pool for j in range(J)]
    return Z, names, K


def stage_base(variant, jmax, tol, ridge=0.0, quiet=False, cnorm="pooled",
               poolname="rich", use_whiten=True, train_ros=None):
    pool = POOLS[poolname]
    cases = load_cases(variant, cnorm)
    W, lam, J = whiten(cases, jmax, tol, use_whiten)
    Z, names, K = _design_U(cases, pool)
    tr = [i for i, c in enumerate(cases)
          if (train_ros is None or c["ro"] in train_ros)]
    Zall = np.vstack([Z[i] for i in tr])
    yall = np.concatenate([cases[i]["b23"] for i in tr])
    wall = np.concatenate([np.full(cases[i]["Nv"],
                                   1.0 / (cases[i]["Nv"] * cases[i]["var_c"]))
                           for i in tr])
    coef = wlstsq(Zall, yall, wall, ridge)
    preds = [Z[i] @ coef for i in range(len(cases))]
    r = eval_nmse(preds, cases)
    rtr = eval_nmse([preds[i] for i in tr], [cases[i] for i in tr])
    te = [i for i, c in enumerate(cases)
          if (train_ros is not None and c["ro"] not in train_ros)]
    rte = (eval_nmse([preds[i] for i in te], [cases[i] for i in te])
           if te else None)
    cmat = coef.reshape(K, J).T                   # cmat[j,k] ↔ φ_k·U_j

    # 每条 β_j 的逐点靶标
    tgt = {}
    for j in range(J):
        tgt[j] = [np.column_stack([fn(c["f"]) for _, fn in pool]) @ cmat[j]
                  for c in cases]

    save = {"variant": variant, "pool": np.array([n for n, _ in pool]),
            "poolname": poolname,
            "train_ros": np.array(train_ros if train_ros else ROS, dtype=float),
            "W": W, "lam": lam, "J": J, "coef": coef, "K": K, "ridge": ridge,
            "cnorm": cnorm, "chan_names": np.array(NAMED[variant])}
    for i, c in enumerate(cases):
        save[f"c{i}_X"] = np.column_stack([c["f"][k] for k in BASE])
        save[f"c{i}_b23"] = c["b23"]
        save[f"c{i}_ro"] = c["ro"]
        save[f"c{i}_var_c"] = c["var_c"]
        save[f"c{i}_U"] = c["U"]
        for j in range(J):
            save[f"c{i}_b{j}"] = tgt[j][i]
    if not quiet:
        np.savez_compressed(_tpaths(variant, train_ros)[0], **save)

    allb = np.concatenate([np.concatenate(tgt[j]) for j in range(J)])
    summary = dict(variant=variant, stage="base", pool=[n for n, _ in pool],
                   poolname=poolname, train_ros=list(train_ros) if train_ros else list(ROS),
                   lam=[float(x) for x in lam], J=J, n_params=int(coef.size),
                   ridge=ridge, cnorm=cnorm, loss_vw=r["loss_vw"], worst=r["worst"],
                   per_case=r["per_case"],
                   loss_train=rtr["loss_vw"], worst_train=rtr["worst"],
                   loss_test=(rte["loss_vw"] if rte else None),
                   worst_test=(rte["worst"] if rte else None),
                   coef=[[float(v) for v in cmat[j]] for j in range(J)],
                   beta_rms=[float(np.sqrt(np.mean(np.concatenate(tgt[j]) ** 2)))
                             for j in range(J)],
                   beta_absmax=float(np.max(np.abs(allb))),
                   chan_names=NAMED[variant])
    print(f"[basisB2:base:{variant}] J={J} ridge={ridge:g} "
          f"loss={r['loss_vw']:.4f} worst={r['worst']:.4f} "
          f"train={rtr['loss_vw']:.4f}"
          + (f" test={rte['loss_vw']:.4f}" if rte else "")
          + f" β_absmax={summary['beta_absmax']:.3g} | " +
          " ".join(f"{k[2:]}:{v:.3f}" for k, v in r["per_case"].items()), flush=True)
    if not quiet:
        print(f"[basisB2:base:{variant}] β_rms=" +
              " ".join(f"β{j+1}:{summary['beta_rms'][j]:.3g}" for j in range(J)),
              flush=True)
        for j in range(J):
            print(f"   β{j+1}(Θ) = " + " + ".join(
                f"({v:+.4g})·{k}" for v, k in zip(cmat[j], [n for n, _ in pool])),
                flush=True)
    if not quiet:
        json.dump(summary, open(_tpaths(variant, train_ros)[1], "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)
    return summary


# ---------------------------------------------------------------------------
def _cases_for_target(variant, j, train_ros=None):
    d = np.load(_tpaths(variant, train_ros)[0])
    tr = [i for i, ro in enumerate(ROS)
          if (train_ros is None or float(ro) in train_ros)]
    pooled = np.concatenate([d[f"c{i}_b{j}"].astype(float) for i in tr])
    scale = float(np.sqrt(np.mean(pooled ** 2))) or 1.0
    cs = []
    for i in range(len(ROS)):
        X = d[f"c{i}_X"].astype(float)
        b = d[f"c{i}_b{j}"].astype(float) / scale
        Uj = d[f"c{i}_U"].astype(float)[:, j]
        var_c = float(d[f"c{i}_var_c"]) + 1e-300
        nv = X.shape[0]
        w = Uj ** 2 / (nv * var_c)
        cs.append(dict(i=i, ro=float(d[f"c{i}_ro"]), Xfit=X, yfit=b, w=w,
                       Nv=nv))
    return cs, d, scale, tr


PHYS = """旋转槽道流 ASM 张量基闭合 (Re_τ=180, 系统旋转沿流向, Ro_τ=7.5/15/30/75/150).
目标形式: b_ij = Σ_m a_m(Θ)·T̃⁽ᵐ⁾_ij  —— 张量基展开, 系数 a_m 是 Θ 的标量函数.
本次拟合**第 {M} 条系数 a_{M}(Θ)**; T̃⁽ᵐ⁾ 是张量通道 23 分量经**逐工况 RMS 归一**
后的值 (已知, 无需拟合, 量级 O(1)); 目标 a_{M} 也已归一到 |a|≲O(1).

张量通道的物理来源 (据此判断 a_{M} 的角色):
- T1  = S 通道 (平均剪切应变 S23); 对应线性涡黏闭包 b23 = -C·S23. **旋转不直接进入**.
- T5, T6, T10 = W̄/Ω 通道 (内禀旋转 W̄ 与系统旋转 Ω 组合). **这是旋转敏感性的唯一载体**.
推论: a1 是"涡黏型"系数 (主要由 eta2、ret、壁面 yw 决定, 对 roloc 弱依赖);
      a5/a6/a10 是"旋转响应"系数 (强烈依赖 roloc, 常随 roloc 饱和/衰减).

Θ (系数自变量, 逐点局部量; 5 工况):
- eta2 = tr(S²) >= 0     (Pope I1; 1e-6 → 3e2, 近壁极小、对数变化)
- zeta2 = tr(W²) >= 0    (Pope I2; g/e 基中含 Ω 贡献, 旋转信息主载体)
- s2o2 = tr(S²W²)        (Pope 不变量; 2e4 量级)
- ret  = k²/(νε) 局部湍流雷诺数 (6e-5 → 5e4)
- roloc = 2Ωk/ε 局部旋转数 (6e-5 → 15.4; 区分 5 个旋转工况的核心变量)
- yw = 1-|y/h| ∈ [0,1] 真壁距 (0=壁面, 1=槽道中心) — 非局部壁面扩展量

ASM/SSG 闭包已知结构 (请优先据此构造候选):
1. 系数是 Θ 的**光滑函数**; 常数系数跨工况直接崩溃 (NMSE 3-5).
2. 旋转饱和族: 高 roloc 下 a ~ tanh(C·roloc)、C/(C'+roloc)、exp(-C·roloc)、√roloc.
3. 近壁族: eta2/ret 由 1e-6 剧变, 常见 log(eta2+eps)、1/(C+eta2)、eta2/(C+eta2)、
   exp(-C·yw)、yw、yw² 的壁面衰减/增长; 切忌 eta2 的大幂 (会爆).
4. 有理形式 g1 = C1/(1 + C2·eta2 + …) 是 Speziale 类系数的标准形状.
硬规则: 只允许全局常数 C# (5 工况通用), 禁止逐工况常数 K#.

联合线性解的系数结构 (48 参数加权 lstsq 上界, 仅供参考起点, 可自由改进):
{STRUCT}"""

# 知识消融用: 剥离全部 ASM 物理知识的对照提示 (只保留任务定义与变量表)
PHYS_GENERIC = """曲线拟合任务: 拟合一个标量函数 y = f(x1..x6), 数据来自 5 组相关工况
(逐工况 127 点, 已合并). 目标 y 已归一到 |y|≲O(1).

自变量 (逐点):
- x1 = eta2, x2 = zeta2, x3 = s2o2, x4 = ret, x5 = roloc, x6 = yw
(具体物理含义与量纲从略; 量级: eta2 1e-6→3e2, zeta2 同量级, s2o2 ~2e4,
 ret 6e-5→5e4, roloc 6e-5→15.4, yw ∈[0,1]).

要求:
1. f 是光滑函数, 5 组工况须用同一表达式.
2. 硬规则: 只允许全局常数 C#, 禁止逐工况常数 K#."""


def structure_hint(variant):
    """从 base json 读联合解系数, 生成 'β_j(Θ) ≈ ...' 提示文本 (每行前 5 显著项)."""
    try:
        d = json.load(open(OUT / f"basisB2_base_{variant}.json", encoding="utf-8"))
        pool, coef = d["pool"], d["coef"]
        lines = []
        for j, row in enumerate(coef):
            terms = sorted(zip(pool, row), key=lambda kv: -abs(float(kv[1])))[:5]
            body = " + ".join(f"({float(v):+.3g})·{k}" for k, v in terms
                              if abs(float(v)) > 1e-4)
            lines.append(f"b{j + 1}(Θ) ≈ {body}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        print(f"[basisB2] structure_hint WARN: {e}", flush=True)
        return "(结构提示不可用)"


def get_prompt(kind, struct=""):
    p = PHYS if kind == "physics" else PHYS_GENERIC
    return p.replace("{STRUCT}", struct or "(未提供)")


def _compile_expr(dsl):
    """把 DSL/sympy 字符串编译为 fn(*X_cols, *params).

    - LGF 公式: 参数为显式 C<k> 符号, 初值取自 result.json 的 theta (调用方给).
    - PySR 公式: 数值常量内联 → 逐个 Float atom 替换为符号 C9xx (初值=原数值),
      使 PySR 与 LGF 在联合重拟合中同场抛光.
    返回 (fn, param_names, theta0).
    """
    import sympy as sp
    from scipy.optimize import least_squares as _lsq  # noqa: F401 (占位避免误删)

    s = dsl.replace("^", "**")
    sy = {n: sp.Symbol(n) for n in BASE}
    expr = sp.sympify(s, locals=sy)
    repl, theta0, names = [], [], []
    for k, v in enumerate(sorted(expr.atoms(sp.Float), key=str)):
        nm = f"C9{90 + k}"
        repl.append((v, sp.Symbol(nm)))
        names.append(nm)
        theta0.append(float(v))
    if repl:
        expr = expr.xreplace(dict(repl))
    free = sorted({str(v) for v in expr.free_symbols if str(v).startswith("C")},
                  key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    syms = [sy[n] for n in BASE] + [sp.Symbol(c) for c in free]
    fn = sp.lambdify(syms, expr, printer=None,
                     modules=[{"inv": lambda x: 1.0 / x}, "numpy"])
    # LGF: theta0 由调用方按 result.json theta 填; PySR: 用内联值
    th0 = [dict(zip(names, theta0)).get(c, 0.0) for c in free]
    return fn, free, th0


def _eval_a(fn, X, th):
    with np.errstate(all="ignore"):
        v = fn(*[X[:, i] for i in range(X.shape[1])], *th)
    v = np.asarray(v, dtype=float)
    if v.shape != (X.shape[0],):        # 常数表达式广播
        v = np.broadcast_to(v, (X.shape[0],)).copy()
    return np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)


def joint_refit(d0, rows, cases, scales, J, train_ros=None):
    """固定 SR 公式形状, 把 J 条公式的全部参数联合最小化重组 NMSE.

    返回 refit dict (含 loss_vw/worst/per_case/theta/loss_train/loss_test),
    失败返回 None.
    """
    try:
        fns, prs, th0 = [], [], []
        for j in range(J):
            fn, params, th0_p = _compile_expr(rows[j]["dsl"])
            # LGF: 优先用 result.json 里的 theta 作为初值
            th = rows[j].get("theta")
            if th and len(th) >= len(params):
                th0_p = [float(x) for x in th[:len(params)]]
            fns.append(fn)
            prs.append(params)
            th0.extend(th0_p)
        seg, o = [], 0
        for j in range(J):
            seg.append((o, o + len(prs[j])))
            o += len(prs[j])
        if o == 0:
            return None
        th0 = np.asarray(th0, dtype=float)

        tr = [i for i, c in enumerate(cases)
              if (train_ros is None or c["ro"] in train_ros)]

        def recon(th, c, i):
            X = d0[f"c{i}_X"].astype(float)
            U = d0[f"c{i}_U"].astype(float)
            acc = np.zeros(c["Nv"])
            for j in range(J):
                a = _eval_a(fns[j], X, th[seg[j][0]:seg[j][1]])
                acc = acc + a * scales[j] * U[:, j]
            return acc

        def nmse(th, idx):
            tot, worst = 0.0, 0.0
            per = {}
            for i in idx:
                c = cases[i]
                r = recon(th, c, i) - c["b23"]
                n = float(np.mean(r ** 2) / c["var_c"])
                per[f"Ro{c['ro']:g}"] = n
                tot += n
                worst = max(worst, n)
            return tot / max(len(idx), 1), worst, per

        def resid(th):
            return np.concatenate([
                (recon(th, cases[i], i) - cases[i]["b23"])
                / np.sqrt(cases[i]["var_c"]) for i in tr])

        from scipy.optimize import least_squares
        sol = least_squares(resid, th0, method="lm", max_nfev=20000)
        th1 = np.clip(sol.x, -1e8, 1e8)
        lt, wt, per_tr = nmse(th1, tr)
        te = [i for i in range(len(cases)) if i not in tr]
        la, wa, per = nmse(th1, list(range(len(cases))))
        rte = nmse(th1, te) if te else (None, None, None)
        per.update(per_tr)  # 完整 per_case (train/test 混排展示用)
        return dict(loss_vw=la, worst=wa, per_case=per,
                    theta=[float(v) for v in th1],
                    n_params=int(o), loss_train=lt, worst_train=wt,
                    loss_test=(rte[0] if te else None),
                    worst_test=(rte[1] if te else None))
    except Exception as e:  # noqa: BLE001
        print(f"[basisB2:refit] WARN: {type(e).__name__}: {e}", flush=True)
        return None


def _fit_pysr(cs_fit, cs_pred, seed, niter, maxsize, run_dir, label):
    from pysr import PySRRegressor
    X = np.vstack([c["Xfit"] for c in cs_fit])
    y = np.concatenate([c["yfit"] for c in cs_fit])
    w = np.concatenate([c["w"] for c in cs_fit])
    w = w / max(w.sum(), 1e-300) * len(w)
    unary = ["exp", "log", "sqrt", "tanh"]
    nested = {u: {v: 0 for v in unary} for u in unary}
    mdl = PySRRegressor(
        niterations=niter, populations=8, population_size=40, maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"], unary_operators=unary,
        nested_constraints=nested, constraints={u: 12 for u in unary},
        model_selection="best", random_state=seed, temp_equation_file=True,
        verbosity=0, progress=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mdl.fit(X, y, weights=w, variable_names=BASE)
    el = time.time() - t0
    preds = [np.asarray(mdl.predict(c["Xfit"]), dtype=float) for c in cs_pred]
    eqs = mdl.equations_
    i = int(np.argmin(eqs["loss"].to_numpy()))
    return dict(dsl=str(eqs.iloc[i]["equation"]),
                complexity=int(eqs.iloc[i]["complexity"]),
                loss=float(eqs.iloc[i]["loss"]), elapsed_s=round(el, 1),
                preds=preds)


def _fit_lgf(cs_fit, cs_pred, method, backend, seed, budget, run_dir, label,
             mstr, pop=60, gens=25, maxparams=6, prompt="physics",
             max_edges=16):
    be = BACKENDS[backend]
    api_key = os.environ.get("LGF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if backend != "heuristic" and not api_key:
        raise SystemExit(f"[basisB2] backend={backend} 需 LGF_LLM_API_KEY")
    client = llm_ops.make_client(be["backend"], url=be["url"], model=be["model"],
                                 api_key=api_key, timeout=300.0,
                                 max_tokens=be.get("max_tokens", 8192),
                                 extra_body=be.get("extra_body"))
    s = (zlib.crc32(f"basisB2|{label}".encode()) + seed) % (2 ** 31)
    cfg = LGFConfig(
        population=pop, generations=gens, sub_sample=300, time_budget_s=budget,
        max_edges=max_edges, max_params=maxparams, final_starts=10,
        final_maxiter=200,
        llm_backend=be["backend"], llm_url=be["url"], llm_model=be["model"],
        llm_api_key=api_key, llm_timeout=240.0, seed=s,
        llm_ginit_only=(method == "lgf_go"), wall_seeds=False,
        physics_context=get_prompt(prompt,
                                   structure_hint(label.split("_")[0])
                                   ).replace("{M}", mstr),
        global_params_only=True, ginit_retry=2, ginit_min_kept=3,
        ginit_fewshot=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    lgf_cases = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}", w=c["w"])
                 for c in cs_fit]
    t0 = time.time()
    res = run_lgf(lgf_cases, BASE, cfg, str(run_dir), client,
                  name=f"basisB2_{label}_{method}")
    el = time.time() - t0
    best = res["best"]
    g = lgfdsl.parse_dsl(best["dsl"], BASE)
    cg = CompiledGraph(g, backend="numpy")
    th = np.asarray(best["theta"], dtype=float)
    preds = [np.asarray(cg.eval_case(th, c["Xfit"]), dtype=float)
             for c in cs_pred]
    return dict(dsl=best["dsl"], complexity=best["complexity"],
                loss=float(best["loss"]), theta=[float(v) for v in th],
                elapsed_s=round(el, 1), preds=preds)


def stage_sr(variant, method, backend, seed, budget, niter, maxsize, tag="",
             cnorm="percase", use_whiten=True, train_ros=None, pop=60, gens=25,
             maxparams=6, prompt="physics", max_edges=16):
    d0 = np.load(_tpaths(variant, train_ros)[0])
    J = int(d0["J"])
    rows, per_target, scales = [], {}, {}
    for j in range(J):
        cs, d, scale, tr = _cases_for_target(variant, j, train_ros)
        cs_fit = [cs[i] for i in tr]
        rd = OUT / f"runs_basisB2_{method}" / (f"{variant}_b{j}" +
                                               (f"_s{seed}" if seed else ""))
        if method == "pysr":
            fit = _fit_pysr(cs_fit, cs, seed, niter, maxsize, rd,
                            f"{variant}_b{j}")
        else:
            fit = _fit_lgf(cs_fit, cs, method, backend, seed, budget, rd,
                           f"{variant}_b{j}", str(j + 1), pop, gens, maxparams,
                           prompt, max_edges)
        per_target[j] = fit
        scales[j] = scale
        print(f"[basisB2:sr:{variant}:{method}] b{j}: cplx={fit['complexity']} "
              f"loss_coef={fit['loss']:.3e} {fit['elapsed_s']}s :: {fit['dsl']}",
              flush=True)
        rows.append(dict(target=f"b{j+1}", dsl=fit["dsl"],
                         complexity=fit["complexity"], loss_coef=fit["loss"],
                         scale=scale, theta=fit.get("theta")))

    # 重组 b23 = Σ_j scale_j·sr_j(Θ)·U_j (评估覆盖全部工况)
    cases = load_cases(variant, cnorm)
    preds = []
    for i in range(len(cases)):
        acc = np.zeros(cases[i]["Nv"])
        for j in range(J):
            acc = acc + per_target[j]["preds"][i] * scales[j] * d0[f"c{i}_U"][:, j]
        preds.append(acc)
    r = eval_nmse(preds, cases)
    tr_i = [i for i, c in enumerate(cases)
            if (train_ros is None or c["ro"] in train_ros)]
    te_i = [i for i, c in enumerate(cases) if i not in tr_i]
    rtr = eval_nmse([preds[i] for i in tr_i], [cases[i] for i in tr_i])
    rte = (eval_nmse([preds[i] for i in te_i], [cases[i] for i in te_i])
           if te_i else None)
    print(f"[basisB2:sr:{variant}:{method}] 重组 b23: loss={r['loss_vw']:.4f} "
          f"worst={r['worst']:.4f} | train={rtr['loss_vw']:.4f}"
          + (f" test={rte['loss_vw']:.4f}" if rte else "")
          + " | " + " ".join(f"{k[2:]}:{v:.3f}" for k, v in r["per_case"].items()),
          flush=True)

    # --- 联合常数重拟合: 固定 SR 公式形状, J 条公式的参数联合抛光 ---
    refit = joint_refit(d0, rows, cases, scales, J, train_ros)
    if refit is not None:
        print(f"[basisB2:refit:{variant}:{method}] 联合重拟合: "
              f"loss={refit['loss_vw']:.4f} worst={refit['worst']:.4f} "
              f"(n_par={refit['n_params']}) | "
              + " ".join(f"{k[2:]}:{v:.3f}" for k, v in refit["per_case"].items()),
              flush=True)

    sfx = f"_s{seed}" if (method == "pysr" and seed) else ""
    rd = OUT / "runs_basisB2" / f"{variant}_{method}{sfx}{tag}"
    rd.mkdir(parents=True, exist_ok=True)
    summary = dict(variant=variant, method=method, backend=backend, stage="sr",
                   J=J, loss_vw=r["loss_vw"], worst=r["worst"],
                   per_case=r["per_case"], targets=rows, seed=seed,
                   prompt=prompt, pop=pop, gens=gens, maxparams=maxparams,
                   train_ros=list(train_ros) if train_ros else list(ROS),
                   loss_train=rtr["loss_vw"], worst_train=rtr["worst"],
                   loss_test=(rte["loss_vw"] if rte else None),
                   worst_test=(rte["worst"] if rte else None),
                   refit=refit, chan_names=NAMED[variant])
    json.dump(summary, open(rd / "result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[basisB2:sr:{variant}:{method}] -> {rd/'result.json'}", flush=True)
    return summary


def stage_refit(variant, method, tag="", cnorm="percase", seed=0):
    """读已有 result.json, 重做联合重拟合并回写 (不重跑 SR)."""
    sfx = f"_s{seed}" if (method == "pysr" and seed) else ""
    rd = OUT / "runs_basisB2" / f"{variant}_{method}{sfx}{tag}"
    p = rd / "result.json"
    if not p.exists():
        print(f"[basisB2:refit] SKIP (missing {p})", flush=True)
        return None
    run = json.load(open(p, encoding="utf-8"))
    d0 = np.load(_tpaths(variant, None)[0])
    J = int(run["J"])
    rows = run["targets"]
    scales = {j: float(rows[j]["scale"]) for j in range(J)}
    cases = load_cases(variant, cnorm)
    refit = joint_refit(d0, rows, cases, scales, J, None)
    run["refit"] = refit
    if refit is not None:
        print(f"[basisB2:refit:{variant}:{method}{tag}] "
              f"loss={refit['loss_vw']:.4f} worst={refit['worst']:.4f} "
              f"(n_par={refit['n_params']}, 原 {run['loss_vw']:.4f}) | "
              + " ".join(f"{k[2:]}:{v:.3f}" for k, v in refit["per_case"].items()),
              flush=True)
    json.dump(run, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[basisB2:refit:{variant}:{method}{tag}] -> {p}", flush=True)
    return refit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "sr", "refit"], required=True)
    ap.add_argument("--variant", choices=["v", "g", "e", "m"], required=True)
    ap.add_argument("--method", choices=["pysr", "lgf_go", "lgf_mg"], default="pysr")
    ap.add_argument("--backend", choices=list(BACKENDS), default="glm53")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--pysr_iters", type=int, default=80)
    ap.add_argument("--pysr_maxsize", type=int, default=25)
    ap.add_argument("--jmax", type=int, default=4)
    ap.add_argument("--wtol", type=float, default=0.0)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--cnorm", choices=["pooled", "percase"], default="percase")
    ap.add_argument("--pool", choices=["rich", "compact"], default="rich")
    ap.add_argument("--whiten", type=int, choices=[0, 1], default=0)
    ap.add_argument("--train_ros", default="")
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--maxparams", type=int, default=6)
    ap.add_argument("--maxedges", type=int, default=16)
    ap.add_argument("--prompt", choices=["physics", "generic"], default="physics")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tro = ([float(x) for x in a.train_ros.split(",") if x.strip()]
           if a.train_ros else None)
    if a.stage == "base":
        if a.sweep:
            for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                stage_base(a.variant, a.jmax, a.wtol, lam, quiet=True,
                           cnorm=a.cnorm, poolname=a.pool,
                           use_whiten=bool(a.whiten), train_ros=tro)
        else:
            stage_base(a.variant, a.jmax, a.wtol, a.ridge, cnorm=a.cnorm,
                       poolname=a.pool, use_whiten=bool(a.whiten),
                       train_ros=tro)
    elif a.stage == "refit":
        stage_refit(a.variant, a.method, a.tag, cnorm=a.cnorm, seed=a.seed)
    else:
        stage_sr(a.variant, a.method, a.backend, a.seed, a.budget,
                 a.pysr_iters, a.pysr_maxsize, a.tag, cnorm=a.cnorm,
                 use_whiten=bool(a.whiten), train_ros=tro, pop=a.pop,
                 gens=a.gens, maxparams=a.maxparams, prompt=a.prompt,
                 max_edges=a.maxedges)


if __name__ == "__main__":
    main()
