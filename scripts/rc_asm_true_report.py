# -*- coding: utf-8 -*-
"""rc_asm_true_report.py — 真 ASM 对比报告生成器.

汇总:
  - prep_report_trueasm.json   (基线: linear / cc10 / cc10_pc / oracle4 / oracle10)
  - runs_trueasm_pysr/{v,g,e}/result.json      (PySR 纯 SR)
  - runs_trueasm_lgf/{v,g,e}_{mg,go}/result.json (LGF + MiniMax M2.7)
输出: outputs/asm/ASM_trueasm_report.md

用法: python -X utf8 scripts/rc_asm_true_report.py [--skip_missing]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(REPO, "outputs", "asm")
VARIANTS = ["v", "g", "e"]
VAR_LABEL = {
    "v": "v: W = W̄ (纯涡量)",
    "g": "g: W = W̄ + Ω (Gatski-Speziale)",
    "e": "e: W = W̄ + 2Ω (扩展内禀)",
}
FEATS = [f"T{m}" for m in range(1, 11)] + \
        ["eta2", "zeta2", "s3", "so2", "s2o2", "ret", "roloc"]
T_SET = set(FEATS[:10])
SCAL_SET = set(FEATS[10:])


def _jload(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def feature_usage(formula: str):
    """公式中出现的特征 (词边界匹配, T1 不误配 T10)."""
    used = []
    for f in FEATS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(f)}(?![A-Za-z0-9_])", formula):
            used.append(f)
    return used


def method_row(name, m, formula, cplx):
    """一行方法对比 (m = metrics dict)."""
    if m is None or "loss_asm" not in m:
        return dict(name=name, ok=False)
    return dict(
        name=name, ok=True,
        loss_asm=m["loss_asm"], rho_a=m["rho_a"],
        worst=m["worst_case_nmse"],
        oor=m.get("frac_oor", 0.0), max_abs=m.get("max_abs_pred", float("nan")),
        per_case=m["per_case"], formula=formula, cplx=cplx)


def gather():
    prep = _jload(os.path.join(OUT_DIR, "prep_report_trueasm.json")) or {}
    rows = {}          # (variant, method_name) -> row
    pysr_details = {}  # variant -> result json
    lgf_details = {}
    for v in VARIANTS:
        # ---- PySR ----
        r = _jload(os.path.join(OUT_DIR, "runs_trueasm_pysr", v, "result.json"))
        if r and r.get("best"):
            pysr_details[v] = r
            rows[(v, "PySR 纯SR")] = method_row(
                "PySR 纯SR", r["best"]["metrics"], r["best"]["sympy"],
                r["best"]["complexity"])
        # ---- LGF mg / go ----
        for arm in ("mg", "go"):
            r = _jload(os.path.join(OUT_DIR, "runs_trueasm_lgf",
                                    f"{v}_{arm}", "result.json"))
            if not r:
                continue
            lgf_details[(v, arm)] = r
            best = None
            for e in r.get("entries", []):
                if e.get("tag") == "best":
                    best = e
                    break
            if best is None or "metrics" not in best:
                continue
            name = f"LGF-{arm}" + (" (M_GUIDE)" if arm == "mg" else " (ginit-only)")
            rows[(v, name)] = method_row(
                name, best["metrics"], best["dsl"], best["complexity"])
    return prep, rows, pysr_details, lgf_details


def render(prep, rows, pysr_details, lgf_details):
    L = []
    A = L.append
    A("# 真 ASM (Algebraic Stress Model) 符号回归对比报告")
    A("")
    A("**任务**: Weatheritt & Sandberg (2016 JCP 325:22-37) eq.(17)-(20) 结构的旋转槽道流代数应力模型 —")
    A("单一闭式公式 `b_ij = f(T1..T10, eta2, zeta2, s3, so2, s2o2, ret, roloc)`")
    A("对全部 6 分量 × 5 工况 (Ro_τ=7.5..150, Re_τ=180) 通用, 只允许全局常数 C#.")
    A("")
    A("**两方法严格隔离**: PySR = 纯符号回归 (无 LLM / 无物理提示 / 无 LGF 策略);")
    A("LGF = MiniMax-M2.7 + 物理上下文 (张量基/不变量/Re_t/Ro_loc 含义注入 G_INIT 与 M_GUIDE),")
    A("mg 臂 M_GUIDE 全程开启, go 臂仅 G_INIT. 真 ASM 硬约束: K# 逐工况常数一律 K→C 改写.")
    A("")
    A("**评估**: loss_asm = mean_c(MSE_c/var_c) (行 = 工况×壁面位置×分量);")
    A("ρ_a,case = W&S eq.22 逐点张量余弦对 valid 点平均 (全 3×3 Frobenius);")
    A("作弊检测 = max|pred| 越界占比 (阈值 20×max|b_ref|, V3-c 同款).")
    A("")

    # ---- 1. 基线阶梯 ----
    A("## 1. 基线阶梯 (常数的表达力极限)")
    A("")
    A("| 变体 | linear (1常数) | cc10 (10全局常数) | cc10_pc (逐工况常数) | oracle4 | oracle10 |")
    A("|---|---|---|---|---|---|")
    for v in VARIANTS:
        b = (prep.get(v) or {}).get("baselines", {})
        if not b:
            A(f"| {v} | (缺) | | | | |")
            continue
        A(f"| {VAR_LABEL[v]} | {_fmt(b['linear']['rel_resid'])} "
          f"| {_fmt(b['cc10']['rel_resid'])} | {_fmt(b['cc10_pc']['rel_resid'])} "
          f"| {_fmt(b['oracle4']['rel_resid'])} | {_fmt(b['oracle10']['rel_resid'])} |")
    A("")
    A("- **oracle10 = 0**: 全 10 基张满无迹对称空间 — 难度全在系数函数 g_m(I, Re_t, Ro_loc).")
    A("- **cc10 (全局) < cc10_pc (逐工况)**: 常数版 Pope TBFCM 的工况泛化代价;")
    A("  rot_factor 越大差越大 (v:0.70→e:0.84), 旋转使最优常数逐工况漂移 —")
    A("  这正是 roloc 特征需要捕捉的跨工况差异.")
    A("- s3/so2 结构性恒零 (二维平均流); v 变体 zeta2 ≡ eta2 (‖W̄‖²=‖S‖² 恒等).")
    A("")

    # ---- 2. 主对比表 ----
    A("## 2. 主对比: PySR 纯SR vs LGF+M2.7 (loss_asm / ρ_a)")
    A("")
    for v in VARIANTS:
        A(f"### 变体 {VAR_LABEL[v]}")
        A("")
        A("| 方法 | loss_asm ↓ | ρ_a ↑ | 最差工况 NMSE | 越界占比 | 复杂度 |")
        A("|---|---|---|---|---|---|")
        names = ["PySR 纯SR", "LGF-mg (M_GUIDE)", "LGF-go (ginit-only)"]
        for nm in names:
            r = rows.get((v, nm))
            if not r or not r["ok"]:
                A(f"| {nm} | (未完成) | | | | |")
                continue
            A(f"| {nm} | **{_fmt(r['loss_asm'])}** | **{_fmt(r['rho_a'])}** "
              f"| {_fmt(r['worst'])} | {_fmt(r['oor'], 3)} | {r['cplx']} |")
        b = (prep.get(v) or {}).get("baselines", {})
        if b:
            A(f"| (参考) cc10 全局常数 | {_fmt(b['cc10']['rel_resid'])} | — | — | — | 10常数 |")
        A("")

    # ---- 3. 逐工况明细 ----
    A("## 3. 逐工况 NMSE / ρ_a 明细 (最优公式)")
    A("")
    for v in VARIANTS:
        cand = [(nm, rows[(v, nm)]) for nm in
                ["PySR 纯SR", "LGF-mg (M_GUIDE)", "LGF-go (ginit-only)"]
                if (v, nm) in rows and rows[(v, nm)]["ok"]]
        if not cand:
            continue
        A(f"### 变体 {v}")
        A("")
        for nm, r in cand:
            pcs = " ".join(
                f"Ro{k[2:]}:{_fmt(pc['nmse'], 3)}/ρ{_fmt(pc['rho_a'], 3)}"
                for k, pc in r["per_case"].items())
            A(f"- **{nm}**: {pcs}")
        A("")

    # ---- 4. 最优公式 ----
    A("## 4. 各方法最优公式")
    A("")
    for v in VARIANTS:
        A(f"### 变体 {v}")
        A("")
        for nm in ["PySR 纯SR", "LGF-mg (M_GUIDE)", "LGF-go (ginit-only)"]:
            r = rows.get((v, nm))
            if not r or not r["ok"]:
                continue
            used = feature_usage(r["formula"])
            t_used = [x for x in used if x in T_SET]
            s_used = [x for x in used if x in SCAL_SET]
            A(f"**{nm}** (复杂度 {r['cplx']}, loss_asm={_fmt(r['loss_asm'])}, "
              f"ρ_a={_fmt(r['rho_a'])}):")
            A("")
            A("```")
            A(r["formula"])
            A("```")
            A(f"基张量: {', '.join(t_used) if t_used else '(无 — 纯标量, 警告)'}; "
              f"标量: {', '.join(s_used) if s_used else '(无)'}")
            A("")

    # ---- 5. LLM 通道统计 ----
    A("## 5. LGF LLM 通道统计 (MiniMax-M2.7)")
    A("")
    A("| 臂 | G_INIT 解析 | K→C 改写 | M_GUIDE 调用 | 建议>0 | 应用 | 存活 | M_GUIDE 异常 |")
    A("|---|---|---|---|---|---|---|---|")
    for (v, arm), r in sorted(lgf_details.items()):
        st = r.get("llm_stats", {})
        g = st.get("ginit", {}); mg = st.get("mguide", {})
        A(f"| {v}/{arm} | {g.get('n_parsed', '?')}/{g.get('n_candidates', '?')} "
          f"| {g.get('n_k_rewritten', 0)} | {mg.get('n_calls', 0)} "
          f"| {mg.get('n_nonempty', 0)} | {mg.get('n_applied', 0)} "
          f"| {mg.get('n_alive', 0)} | {(mg.get('last_exception') or '—')[:40]} |")
    A("")

    # ---- 6. 物理解读 ----
    A("## 6. 物理解读")
    A("")
    A("### 6.1 张量基选择 (哪些 T_m 承载各向异性)")
    A("")
    A("| 变体 | 方法 | 使用的张量基 | 标量调制特征 |")
    A("|---|---|---|---|")
    for v in VARIANTS:
        for nm in ["PySR 纯SR", "LGF-mg (M_GUIDE)", "LGF-go (ginit-only)"]:
            r = rows.get((v, nm))
            if not r or not r["ok"]:
                continue
            used = feature_usage(r["formula"])
            t_used = [x for x in used if x in T_SET]
            s_used = [x for x in used if x in SCAL_SET]
            A(f"| {v} | {nm} | {', '.join(t_used) or '—'} | {', '.join(s_used) or '—'} |")
    A("")
    A("### 6.2 结构性备注")
    A("")
    A("- s3/so2 恒零: 公式中把它们当常数发生器 (如 `s3 + 3.31` ≡ 常数) 在本流族合法,")
    A("  但迁移到三维平均流 (s3≠0) 时语义改变 — 部署前需人工审查.")
    A("- v 变体 zeta2 ≡ eta2: 两特征可互换, SR 选其一属正常.")
    A("- ret/roloc 进入系数函数是 Re_t/Ro_local 修正生效的直接证据.")
    A("")

    # ---- 7. 方法学与局限 ----
    A("## 7. 方法学声明与局限")
    A("")
    A("- **复杂度口径不同**: PySR 复杂度=节点数 (maxsize=30); LGF=边数 (max_edges=16) +")
    A("  常数≤6. 两者非严格同预算, 报告按各自帕累托最优呈现.")
    A("- PySR 常数无上限 (BFGS 全局拟合); LGF 常数≤6 且 L-BFGS 多启动.")
    A("- 单 seed 单次运行 (PySR random_state=0, LGF crc32 seed); 无跨 seed 方差估计.")
    A("- 评估在训练数据本身 (无 hold-out): 与 W&S 的 DNS 代数拟合设定一致,")
    A("  但不代表跨流型泛化.")
    A("- LGF 时间预算 800s/臂 vs PySR 50 迭代 (~40s): 计算量不对等,")
    A("  PySR (Julia 多线程) 单位算力远高于 LGF (Python 逐图拟合).")
    A("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_missing", action="store_true")
    args = ap.parse_args()
    prep, rows, pysr_details, lgf_details = gather()
    n_ok = sum(1 for r in rows.values() if r.get("ok"))
    print(f"[report] 收集到 {n_ok} 个方法×变体结果")
    if n_ok == 0 and not args.skip_missing:
        raise SystemExit("没有任何结果 — 先跑 _run_trueasm_campaign.py")
    md = render(prep, rows, pysr_details, lgf_details)
    out = os.path.join(OUT_DIR, "ASM_trueasm_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] -> {out}")


if __name__ == "__main__":
    main()
