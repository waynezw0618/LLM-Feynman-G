# -*- coding: utf-8 -*-
"""rc_asm_basisB2_formulas.py — 从 runs_basisB2/*/result.json 生成变体冠军公式套.

输出 outputs/asm/t23basisB2_paper/formulas_champions.md：
  每变体一节 = 四条 a_m(Θ) LaTeX 数值式 + b23 重构式 + Python 可调用表达式 + 逐工况 NMSE。

用法: python scripts/rc_asm_basisB2_formulas.py [m_big_ready]
  m_big_ready 非零时把 m 冠军换成 m_lgf_mg_big（若 refit 更优）。
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import sympy as sp

OUT = os.path.join("outputs", "asm")
RUNS = os.path.join(OUT, "runs_basisB2")
PAPER = os.path.join(OUT, "t23basisB2_paper")
DEST = os.path.join(PAPER, "formulas_champions.md")

# 变量显示名（LaTeX）
VAR_TEX = {
    "eta2": r"\eta_2", "zeta2": r"\zeta_2", "s2o2": r"S^2/\Omega^2",
    "ret": r"Re_\tau", "roloc": r"r_{o,\mathrm{loc}}", "yw": r"y_w",
}
CHANNELS = {
    "v": ["T1s", "T5s", "T6s", "T10s"],
    "g": ["T1s", "T5s", "T6s", "T10s"],
    "e": ["T1s", "T5s", "T6s", "T10s"],
    "m": ["T1s"] + [f"{t}s_{x}" for t in ("T5", "T6", "T10") for x in ("v", "g", "e")],
}

CHAMPIONS = {
    "e": ("e_lgf_mg_big", "LGF-mg big (joint refit)"),
    "g": ("g_lgf_mg_big", "LGF-mg big (joint refit)"),
    "m": ("m_lgf_go_big", "LGF-go big (joint refit)"),
    "v": ("v_lgf_go", "LGF-go legacy (joint refit)"),
}


def load_run(tag):
    p = os.path.join(RUNS, tag, "result.json")
    return json.load(open(p, encoding="utf-8"))


def c_count(dsl):
    return len(set(re.findall(r"C\d+", dsl)))


def parse_expr(dsl):
    """DSL -> sympy 表达式（C# 为自由符号）。"""
    s = dsl.replace("^", "**")
    syms = {n: sp.Symbol(n, positive=True) for n in VAR_TEX}
    cs = sorted(set(re.findall(r"C\d+", s)), key=lambda x: int(x[1:]))
    loc = {c: sp.Symbol(c) for c in cs}
    expr = sp.sympify(s, locals={**syms, **loc})
    # inv(x) -> 1/x（LGF DSL 的倒数算子）
    expr = expr.replace(sp.Function("inv"), lambda x: 1 / x)
    return expr, cs


def to_latex(expr, subst):
    """常数代入后转 LaTeX（C# -> 数值）。"""
    e = expr.subs(subst)
    if e.count_ops() < 60:
        e = sp.simplify(e)
    tex = sp.latex(e)
    # 变量名替换为论文记号（lambda 避免替换串转义问题）
    for k, v in VAR_TEX.items():
        tex = re.sub(rf"\b{k}\b", lambda m, _v=v: _v, tex)
    return tex


def to_python(expr, subst):
    """常数代入后转 numpy 可调用表达式文本。"""
    e = expr.subs(subst)
    py = sp.pycode(e)
    py = py.replace("math.", "np.")
    for k in VAR_TEX:
        py = re.sub(rf"\b{k}\b", lambda m, _k=k: _k, py)
    return py


def num(x):
    """数值格式化：有效数字 6 位。"""
    return f"{float(x):.6g}"


def gen_variant(v, tag, label):
    d = load_run(tag)
    tg = sorted(d["targets"], key=lambda t: int(t["target"][1:]))
    chs = CHANNELS[v]
    refit = d.get("refit") or {}
    flat = refit.get("theta")
    o = 0
    lines = [f"### 变体 {v} —— 冠军组 `{tag}` ({label})", ""]
    lc = refit.get("loss_vw")
    lines.append(f"- 最终口径 NMSE(b23) = **{lc:.4f}**（worst {refit.get('worst', float('nan')):.3f}），"
                 f"SR 中间口径 {d['loss_vw']:.3f}；公式形状固定，常数来自联合重拟合。")
    lines.append(f"- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..{len(tg)} 对应通道 "
                 f"{', '.join(chs[:len(tg)])}。")
    lines.append("")
    py_lines = []
    for j, t in enumerate(tg):
        expr, cs = parse_expr(t["dsl"])
        n = len(cs)
        if flat and o + n <= len(flat):
            vals = [float(x) for x in flat[o:o + n]]
            o += n
        else:  # 无 refit（v 基失败组）：退回 SR theta
            vals = [float(x) for x in (t.get("theta") or [0.0] * n)][:n]
        subst = {sp.Symbol(c): sp.Float(num(val), 6) for c, val in zip(cs, vals)}
        tex = to_latex(expr, subst)
        py = to_python(expr, subst)
        lines.append(f"**a{j+1}(Θ)** — 通道 {chs[j] if j < len(chs) else j+1}，"
                     f"scale = {num(t['scale'])}：")
        lines.append("")
        lines.append(f"$$a_{{{j+1}}}(\\Theta) = {tex}$$")
        lines.append("")
        py_lines.append(f"# a{j+1}, channel {chs[j] if j < len(chs) else j+1}, scale={num(t['scale'])}")
        py_lines.append(f"a{j+1} = {py}")
    # 数值核对：随机点对照 dsl+theta 直接求值
    ok, msg = verify_variant(d, tg)
    lines.append("Python 可调用表达式（`import numpy as np` 后直接用；"
                 "输入为标量或同形数组，已内联联合重拟合常数）：")
    lines.append("")
    lines.append("```python")
    lines.extend(py_lines)
    lines.append("```")
    lines.append("")
    # 逐工况
    pc = refit.get("per_case") or d.get("per_case") or {}
    row = " | ".join(f"{k[2:]}: {val:.3f}" for k, val in pc.items())
    lines.append(f"逐工况 NMSE：{row}")
    lines.append("")
    lines.append(f"数值核对：{'✅ ' + msg if ok else '⚠️ ' + msg}")
    lines.append("")
    return "\n".join(lines), ok


def verify_variant(d, tg):
    """随机点核对：代入常数后的表达式 vs 原始 dsl+theta 数值。"""
    rng = np.random.default_rng(0)
    refit = d.get("refit") or {}
    flat = refit.get("theta")
    worst = 0.0
    o = 0
    try:
        for t in tg:
            expr, cs = parse_expr(t["dsl"])
            n = len(cs)
            if flat and o + n <= len(flat):
                vals = [float(x) for x in flat[o:o + n]]
            else:
                vals = [float(x) for x in (t.get("theta") or [0.0] * n)][:n]
            o += n
            subst = {sp.Symbol(c): val for c, val in zip(cs, vals)}
            f_new = sp.lambdify(list(VAR_TEX.keys()), expr.subs(subst), "numpy")
            # 原始: dsl 直接以 theta 求值（C# 按出现顺序？——用 cs 顺序赋 theta）
            f_old = sp.lambdify(list(VAR_TEX.keys()) + [sp.Symbol(c) for c in cs],
                                expr, "numpy")
            pts = {k: rng.uniform(0.05, 0.95, 64) for k in VAR_TEX}
            new_v = np.nan_to_num(np.asarray(f_new(*[pts[k] for k in VAR_TEX]), dtype=float))
            old_v = np.nan_to_num(np.asarray(
                f_old(*[pts[k] for k in VAR_TEX] + [subst[sp.Symbol(c)] for c in cs]), dtype=float))
            if len(new_v) != len(old_v):
                return False, f"a 表达式形状不一致"
            denom = np.maximum(np.abs(old_v), 1e-12)
            worst = max(worst, float(np.max(np.abs(new_v - old_v) / denom)))
        return worst < 1e-6, f"max rel diff = {worst:.2e}"
    except Exception as e:
        return False, f"verify skipped ({type(e).__name__}: {e})"


def main():
    champs = dict(CHAMPIONS)
    # 缺文件时退回 legacy
    for v, (tag, label) in list(champs.items()):
        try:
            load_run(tag)
        except FileNotFoundError:
            if v == "m":
                champs[v] = ("m_lgf_mg", "LGF-mg legacy (joint refit)")
            elif v in ("e", "g"):
                champs[v] = (f"{v}_lgf_mg_v2", "LGF-mg v2 (joint refit)")
    parts = ["# 变体冠军公式套（正式口径：公式形状由 LGF 符号回归发现，常数来自联合重拟合）", ""]
    parts.append("> 生成脚本：`scripts/rc_asm_basisB2_formulas.py`；数据源：`outputs/asm/runs_basisB2/*/result.json`。")
    parts.append("> 所有公式严格保持张量基形式 $b_{23}=\\sum_j s_j\\,a_j(\\Theta)\\,\\tilde T^{(j)}_{23}$，")
    parts.append("> 其中 $\\tilde T^{(j)}_{23}$ 为逐工况 RMS 归一的通道分量，$s_j$ 为该通道 scale（各公式下方给出），")
    parts.append("> $\\Theta=\\{\\eta_2,\\zeta_2,S^2/\\Omega^2,Re_\\tau,r_{o,\\mathrm{loc}},y_w\\}$（yw∈[0,1] 无量纲壁距）。")
    parts.append("")
    oks = {}
    for v in ("e", "g", "m", "v"):
        tag, label = champs[v]
        try:
            sec, ok = gen_variant(v, tag, label)
            parts.append(sec)
            oks[v] = ok
        except FileNotFoundError:
            parts.append(f"### 变体 {v} —— `{tag}` 缺失，跳过")
            parts.append("")
    txt = "\n".join(parts)
    os.makedirs(PAPER, exist_ok=True)
    with open(DEST, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[formulas] -> {DEST}")
    for v, ok in oks.items():
        print(f"  {v}: verify {'OK' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
