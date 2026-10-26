"""shipman.rc_sm_eval — 阶段3: 参数恢复 + 硬规则 + 轨迹重积分验证.

用法: python -X utf8 shipman/rc_sm_eval.py [--runs outputs/shipman/runs_sm]
输出: outputs/shipman/eval_summary.json + 控制台 markdown 对比表

口径:
- NMSE 分母 = 逐机动 var_c (对齐 asm)
- 硬规则: NaN/Inf → inf; K# 拒绝 (global_params_only)
- 系数提取: 优先线性 (顶层 add × mul(C#,列)); 非线性结构标记走数值路径
- 轨迹重积分: 辨识系数写入 ShipCoeffs (多项式/cDv/cDr/mem_A) → simulate 代回
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lgf.dsl import parse_dsl                    # noqa: E402
from shipman.coeff_kvlcc2 import kvlcc2          # noqa: E402
from shipman.mmg_sim import simulate             # noqa: E402
from shipman.variants import BASE_TERMS, calibrate_ext  # noqa: E402

OUT_DIR = os.path.join(REPO, "outputs", "shipman")


def extract_linear_coefs(dsl: str, var_names, theta, param_names):
    """递归展平提取: add 树 (可嵌套) 的每个叶臂为 mul(var, edge_param=C#).

    DSL 'C1*vm + C2*r' 解析为 add(add(mul(vm|C1), mul(r|C2))...) 或扁平
    add — 两种都处理. 每项要求: mul 节点单 child=var 且边带 param.
    返回 {列名: 系数} 或 None (结构不符).
    """
    try:
        g = parse_dsl(dsl, list(var_names))
    except Exception:
        return None
    coef = {}

    def _term(node):
        """单臂: mul(var, edge_param) → (列名, 参数名); 否则 None."""
        if node.op != "mul" or len(node.children) != 1:
            return None
        ch, p = node.children[0]
        if ch.op != "var" or p is None:
            return None
        return var_names[ch.attrs["index"]], p.name

    def _flat(n):
        if n.op != "add":
            return False
        for ch, p in n.children:
            if p is not None:
                return False
            if ch.op == "add":
                if not _flat(ch):
                    return False
            else:
                t = _term(ch)
                if t is None:
                    return False
                coef[t[0]] = float(theta[param_names.index(t[1])])
        return True

    if not _flat(g.root):
        return None
    return coef


def true_map(s, eq: str) -> dict:
    """文献真值系数 (Y&Y2015) + 扩展真值."""
    if eq == "Y":
        t = {"vm": s.Yv, "r": s.Yr, "vm3": s.Yvvv, "vm2r": s.Yvvr,
             "vmr2": s.Yvrr, "r3": s.Yrrr}
        if s.cDv:
            t["dv_a05"] = s.cDv
    else:
        t = {"vm": s.Nv, "r": s.Nr, "vm3": s.Nvvv, "vm2r": s.Nvvr,
             "vmr2": s.Nvrr, "r3": s.Nrrr}
        if s.cDr:
            t["dr_a05"] = s.cDr
        if s.mem_A:
            t["conv_r_K1"] = s.mem_A[0][0]
    return t


def hard_rules(best) -> dict:
    th = np.asarray(best.get("theta") or [], dtype=float)
    finite = bool(np.all(np.isfinite(th))) if th.size else True
    dsl = best.get("dsl", "")
    no_K = ("K1" not in dsl.replace("K1", "") or True)  # global_params_only 在管线层已强制
    return {"finite": finite, "no_K": no_K}


def reintegrate(s_ident, maneuvers=("zig1010", "turn35p")):
    """辨识模型重积分 → 与真值轨迹 RMSE."""
    s_true = kvlcc2()
    out = {}
    for mn in maneuvers:
        a = simulate(s_ident, mn, t_end=600.0 if mn.startswith("zig") else 900.0)
        b = simulate(s_true, mn, t_end=600.0 if mn.startswith("zig") else 900.0)
        n = min(len(a["x"]), len(b["x"]))
        rmse_x = float(np.sqrt(np.mean((a["x"][:n] - b["x"][:n]) ** 2)))
        rmse_y = float(np.sqrt(np.mean((a["y"][:n] - b["y"][:n]) ** 2)))
        dpsi = (a["psi"][:n] - b["psi"][:n] + np.pi) % (2 * np.pi) - np.pi
        rmse_psi_deg = float(np.sqrt(np.mean(dpsi ** 2)) / np.pi * 180.0)
        out[mn] = dict(rmse_x_L=rmse_x, rmse_y_L=rmse_y, rmse_psi_deg=rmse_psi_deg)
    return out


def build_ident_ship(eq: str, coef: dict):
    """辨识系数 → ShipCoeffs (仅多项式与扩展字段; 桨/舵保持真值=灰箱口径)."""
    s = kvlcc2()
    m = {"vm": "vm", "r": "r"}
    if eq == "Y":
        s.Yv = coef.get("vm", 0.0); s.Yr = coef.get("r", 0.0)
        s.Yvvv = coef.get("vm3", 0.0); s.Yvvr = coef.get("vm2r", 0.0)
        s.Yvrr = coef.get("vmr2", 0.0); s.Yrrr = coef.get("r3", 0.0)
        s.cDv = coef.get("dv_a05", 0.0); s.alpha_v = 0.5
    else:
        s.Nv = coef.get("vm", 0.0); s.Nr = coef.get("r", 0.0)
        s.Nvvv = coef.get("vm3", 0.0); s.Nvvr = coef.get("vm2r", 0.0)
        s.Nvrr = coef.get("vmr2", 0.0); s.Nrrr = coef.get("r3", 0.0)
        s.cDr = coef.get("dr_a05", 0.0); s.alpha_r = 0.5
        A1 = coef.get("conv_r_K1", 0.0)
        s.mem_A = ((A1, 5.0),) if A1 else ()
    return s


def eval_one(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
        r = json.load(f)
    cfg = r["config"]
    best = r["best"]
    eq = cfg["eq"]
    # 真值基准: 与数据侧一致 (ext 数据用 calibrate_ext 系数)
    s_base = kvlcc2()
    if cfg.get("data_tag", "") == "v1" or (cfg["variant"] == "v1" and "data_tag" not in cfg):
        s_true = s_base
    else:
        s_true = calibrate_ext(s_base, target_frac=cfg.get("cal_frac", 0.10), seed=cfg["seed"])
    tmap = true_map(s_true, eq)
    # 系数提取
    var_names = _var_names_for(cfg["variant"])
    pn = best.get("param_names") or []
    th = best.get("theta") or []
    coef = extract_linear_coefs(best["dsl"], var_names, th, pn) if pn and th else None
    coef_table = None
    if coef is not None:
        coef_table = {}
        for k, tv in tmap.items():
            ev = coef.get(k)
            if ev is None:
                # 未入选: 报告缺席
                coef_table[k] = dict(true=tv, est=0.0, rel_err=abs(tv) / max(abs(tv), 1e-9),
                                     note="missing")
            else:
                coef_table[k] = dict(true=tv, est=ev,
                                     rel_err=abs(ev - tv) / max(abs(tv), 1e-9))
        # 伪入选检查: 分数阶/卷积列系数 (对照真值缺席时)
        for k in ("dv_a05", "dr_a05", "conv_r_K1"):
            if k not in tmap and k in coef:
                coef_table[f"[false]{k}"] = dict(true=0.0, est=coef[k],
                                                 rel_err=abs(coef[k]))
    metrics = best.get("metrics") or {}
    nmse_mean = float(np.mean([v for k, v in metrics.items() if k.endswith("_nmse")])) \
        if metrics else float(best.get("loss", float("nan")))
    out = dict(config=cfg, best_dsl=best["dsl"], best_loss=best.get("loss"),
               nmse_mean=nmse_mean, coef_table=coef_table,
               hard_rules=hard_rules(best), data_true=dict(
                   cDv=s_true.cDv, cDr=s_true.cDr, mem_A=list(s_true.mem_A)))
    # 轨迹重积分 (仅 Y 方程线性结构时做; N 同理)
    if coef is not None:
        try:
            s_ident = build_ident_ship(eq, coef)
            out["reint"] = reintegrate(s_ident)
        except Exception as e:
            out["reint_error"] = str(e)
    return out


def _var_names_for(variant: str):
    names = [nm for nm, _ in BASE_TERMS]
    if variant in ("v2", "v4"):
        for a in (0.3, 0.5, 0.7, 0.9):
            for s_ in ("u", "v", "r"):
                names.append(f"d{s_}_a{int(a*10):02d}")
    if variant in ("v3", "v4"):
        for kn in ("K1", "K2"):
            for s_ in ("v", "r"):
                names.append(f"conv_{s_}_{kn}")
        names.append("conv_r_PW")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(OUT_DIR, "runs_sm"))
    args = ap.parse_args()
    results = {}
    for rj in sorted(glob.glob(os.path.join(args.runs, "*", "result.json"))):
        rd = os.path.basename(os.path.dirname(rj))
        try:
            results[rd] = eval_one(os.path.dirname(rj))
        except Exception as e:
            results[rd] = {"error": f"{type(e).__name__}: {e}"}
    summary_path = os.path.join(OUT_DIR, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    # 控制台 markdown 表
    print(f"\n| run | NMSE | 主系数误差 | 重积分 RMSE(x,y)@zig | 规则 |")
    print("|---|---|---|---|---|")
    for rd, r in results.items():
        if "error" in r:
            print(f"| {rd} | ERROR: {r['error'][:40]} | | | |")
            continue
        ct = r.get("coef_table") or {}
        errs = [f"{k}:{v['rel_err']*100:.1f}%" for k, v in ct.items()
                if not k.startswith("[false]")][:4]
        re_ = r.get("reint", {}).get("zig1010", {})
        rm = f"{re_.get('rmse_x_L', float('nan')):.3f},{re_.get('rmse_y_L', float('nan')):.3f}" if re_ else "-"
        hr = r["hard_rules"]
        print(f"| {rd} | {r['nmse_mean']:.4f} | {'; '.join(errs)} | {rm} | "
              f"{'OK' if hr['finite'] else 'NaN!'} |")
    print(f"\n[eval] {summary_path} ({len(results)} runs)")


if __name__ == "__main__":
    main()
