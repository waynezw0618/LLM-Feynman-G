# -*- coding: utf-8 -*-
"""rc_asm_true_eval.py — 真 ASM 统一评估模块 (PySR / LGF 共用).

评估口径 (2026-10-09, 与 Weatheritt & Sandberg 2016 JCP 325:22-37 对齐):
  loss_asm  = mean_c( MSE_c / var_c )                    — 项目主口径 NMSE
              c 遍历 5 个 Ro 工况; MSE_c/var_c 在该工况全部 6 分量
              行 (6Nv) 上计算 (行 = 分量*Nv + 壁面点, 分量优先)
  rho_a,c   = <A_mod(y), A_ref(y)>_F / (|A_mod|_F |A_ref|_F) 逐点余弦
              再对 valid 壁面点取平均 (W&S eq.22 的离散实现)
              A 为由 b6=[11,22,33,12,13,23] 重建的全 3x3 对称张量;
              Frobenius 内积 = 对角项 + 2*非对角项 (与 W&S 一致)
  rho_a     = mean_c(rho_a,c)

作 弊 检 测 (v3 验收 5 硬规则的真 ASM 版):
  max_abs_pred   预测最大幅值 (>20x max|b_ref| 即爆值)
  frac_oor       |pred| > 20*max|b_ref| 的行占比 (V3-c 值域护栏同款阈值)

用法 (作为库):
  from rc_asm_true_eval import load_variant, eval_predictions, eval_case
  data = load_variant("v")            # 5 工况拟合行数据
  m = eval_predictions(preds, data)   # preds: list[np.ndarray(6Nv,)] 每工况
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(REPO, "outputs", "asm")
RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
COMP6 = ["uu", "vv", "ww", "uv", "uw", "vw"]      # [11,22,33,12,13,23]
OOR_FACTOR = 20.0                                  # V3-c 值域护栏同款


def load_variant(variant: str):
    """读 dataset_trueasm_{variant}.npz → 每工况 dict(Xfit, yfit, b6v, yh, Nv, ro)."""
    ds = np.load(os.path.join(OUT_DIR, f"dataset_trueasm_{variant}.npz"))
    cases = []
    for i, ro in enumerate(RO_CASES):
        p = f"c{i}_"
        valid = ds[p + "valid"]
        Xfit = ds[p + "Xfit"]
        yfit = ds[p + "yfit"]
        Nv = int(valid.sum())
        assert yfit.shape[0] == 6 * Nv, f"case{i}: {yfit.shape[0]} != 6*{Nv}"
        cases.append(dict(
            i=i, ro=float(ro), Xfit=Xfit, yfit=yfit, Nv=Nv,
            b6v=ds[p + "b6"][valid].astype(float),
            yh=ds[p + "yh"][valid].astype(float),
            var_c=float(np.var(yfit)),
            max_abs_ref=float(np.max(np.abs(yfit)))))
    ds.close()
    return cases


def _full_tensor(b6: np.ndarray) -> np.ndarray:
    """(N,6)=[11,22,33,12,13,23] → (N,3,3) 对称张量."""
    N = b6.shape[0]
    A = np.zeros((N, 3, 3))
    A[:, 0, 0] = b6[:, 0]; A[:, 1, 1] = b6[:, 1]; A[:, 2, 2] = b6[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = b6[:, 3]
    A[:, 0, 2] = A[:, 2, 0] = b6[:, 4]
    A[:, 1, 2] = A[:, 2, 1] = b6[:, 5]
    return A


def rho_a_profile(pred6: np.ndarray, ref6: np.ndarray) -> np.ndarray:
    """逐点张量余弦 (W&S eq.22 被积函数). pred6/ref6: (N,6)."""
    P = _full_tensor(pred6); R = _full_tensor(ref6)
    N = len(P)
    num = np.einsum("nij,nij->n", P, R)
    den = (np.linalg.norm(P.reshape(N, -1), axis=1)
           * np.linalg.norm(R.reshape(N, -1), axis=1))
    out = np.where(den > 1e-300, num / np.maximum(den, 1e-300), 0.0)
    return np.clip(out, -1.0, 1.0)


def eval_case(pred_rows: np.ndarray, case: dict) -> dict:
    """单工况评估. pred_rows: (6Nv,) 分量优先行展开 (index = comp*Nv + j)."""
    Nv = case["Nv"]
    pred_rows = np.asarray(pred_rows, dtype=float).reshape(-1)
    if pred_rows.shape[0] != 6 * Nv:
        raise ValueError(f"pred len {pred_rows.shape[0]} != 6*{Nv}")
    if not np.all(np.isfinite(pred_rows)):
        return dict(nmse=float("inf"), rho_a=-1.0, per_comp={c: float("inf") for c in COMP6},
                    max_abs_pred=float("inf"), frac_oor=1.0)
    pred6 = pred_rows.reshape(6, Nv).T                 # (Nv, 6)
    ref6 = case["b6v"]
    mse = float(np.mean((pred6 - ref6) ** 2))
    nmse = mse / (case["var_c"] + 1e-300)
    per_comp = {}
    for c, name in enumerate(COMP6):
        pc = pred6[:, c]; rc = ref6[:, c]
        per_comp[name] = float(np.mean((pc - rc) ** 2) / (np.var(rc) + 1e-300))
    rho = rho_a_profile(pred6, ref6)
    thr = OOR_FACTOR * (case["max_abs_ref"] + 1e-12)
    return dict(
        nmse=float(nmse), rho_a=float(np.mean(rho)),
        per_comp=per_comp,
        max_abs_pred=float(np.max(np.abs(pred6))),
        frac_oor=float(np.mean(np.abs(pred6) > thr)))


def eval_predictions(preds: list, data: list) -> dict:
    """5 工况联合评估 → 汇总 dict."""
    per_case = []
    for pred, case in zip(preds, data):
        m = eval_case(pred, case)
        m["ro"] = case["ro"]
        per_case.append(m)
    nmses = [m["nmse"] for m in per_case]
    rhos = [m["rho_a"] for m in per_case]
    return dict(
        loss_asm=float(np.mean(nmses)),
        rho_a=float(np.mean(rhos)),
        per_case={f"Ro{m['ro']:g}": dict(nmse=m["nmse"], rho_a=m["rho_a"],
                                         max_abs_pred=m["max_abs_pred"],
                                         frac_oor=m["frac_oor"],
                                         per_comp=m["per_comp"]) for m in per_case},
        worst_case_nmse=float(np.max(nmses)),
        max_abs_pred=float(np.max([m["max_abs_pred"] for m in per_case])),
        frac_oor=float(np.max([m["frac_oor"] for m in per_case])))


def summarize(metrics: dict) -> str:
    """一行摘要 (打印用)."""
    pc = " ".join(f"Ro{k[2:]}:{v['nmse']:.4f}/ρ{v['rho_a']:+.3f}"
                  for k, v in metrics["per_case"].items())
    return (f"loss_asm={metrics['loss_asm']:.4f} rho_a={metrics['rho_a']:+.4f} "
            f"worst={metrics['worst_case_nmse']:.4f} | {pc}")


# ============================================================================
# LGF DSL → 预测 (供评估脚本复用)
# ============================================================================
def lgf_predict(dsl: str, theta, data: list):
    """重编译 DSL + theta → 每工况预测行. 要求 n_c=0 (真 ASM 全局常数)."""
    from lgf import dsl as lgfdsl
    from lgf.compiler import CompiledGraph
    var_names = [str(x) for x in np.load(
        os.path.join(OUT_DIR, "dataset_trueasm_v.npz"))["feat_names"]]
    g = lgfdsl.parse_dsl(dsl, var_names)
    if g.case_names():
        raise ValueError(f"DSL 含逐工况常数 K: {g.case_names()} (真 ASM 禁止)")
    cg = CompiledGraph(g, backend="numpy")
    theta = np.asarray(theta, dtype=float)
    preds = []
    for case in data:
        preds.append(np.asarray(cg.eval_case(theta, case["Xfit"]), dtype=float))
    return preds, var_names


# ============================================================================
# CLI: 评估一个 runs 目录里的 LGF final.json 或 PySR 结果 json
# ============================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json", help="LGF final.json 或 PySR result json 路径")
    ap.add_argument("--variant", default=None, help="数据集变体 (默认从 json 读)")
    ap.add_argument("--key", default=None,
                    help="json 内公式条目键 (LGF: best/top_final[i]; PySR: best)")
    args = ap.parse_args()
    with open(args.run_json, encoding="utf-8") as f:
        j = json.load(f)
    variant = args.variant or j.get("variant", "v")
    data = load_variant(variant)
    if "dsl" in j:                                   # LGF final.json (单条)
        entry = j
    elif "best" in j and "dsl" in j.get("best", {}):  # LGF final.json 完整
        entry = j["best"]
    elif "preds_file" in j:                          # PySR result json
        preds = list(np.load(j["preds_file"])["arr_0"])
        m = eval_predictions(preds, data)
        print(summarize(m))
        return
    else:
        raise SystemExit("无法识别的 json 结构")
    preds, _ = lgf_predict(entry["dsl"], entry["theta"], data)
    m = eval_predictions(preds, data)
    print(summarize(m))
    out = args.run_json.rsplit(".", 1)[0] + "_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
