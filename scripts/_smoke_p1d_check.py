# -*- coding: utf-8 -*-
"""P1d 冒烟: 验证 attribution 新增的 per_case_summary / target_semantics +
M_GUIDE prompt 硬规则块渲染 (M_GUIDE 异常在 pipeline 里被静默吞, 必须显式测)."""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from lgf import attribution, dsl as lgfdsl, llm_ops
from lgf.graph import walk
from rc_asm_run import build_cases

ds = np.load(os.path.join(REPO, "outputs", "asm", "dataset_pope_gsonly.npz"))
final = json.load(open(os.path.join(REPO, "outputs", "asm", "runs_smoke_p1d",
                                    "B_a_T5g_final.json"), encoding="utf-8"))
var_names = [str(x) for x in ds["featB_keys"]]
sel_names = [str(x) for x in ds["sel_basis"]]

g = lgfdsl.parse_dsl(final["best"]["dsl"], var_names)
theta = np.asarray(final["best"]["theta"], dtype=float)
cs = build_cases(ds, "B", "T5g", sel_names, fitness="stress", fit_mask="valid_fit",
                 wall_boost=8.0, comp_w=dict(uu=1, vv=1, ww=1, uv=1, uw=3, vw=5))

# 1) A1/A2: analyze 扩展报告
rep = attribution.analyze(g, theta, cs, var_names)
pc = rep["per_case_summary"]
ts = rep["target_semantics"]
assert len(pc) == 5, f"per_case 应 5 工况, 得 {len(pc)}"
assert all("nmse" in p and "label" in p for p in pc), "per_case 字段缺失"
assert "y_min" in ts and "y_max" in ts and "frac_neg" in ts, "target_semantics 字段缺失"
print("[A1] per_case:", [(p["name"], round(p["nmse"], 2), p["label"]) for p in pc])
print("[A2] y range: [%.3g, %.3g], frac_neg=%.3f" % (ts["y_min"], ts["y_max"], ts["frac_neg"]))
print("[A2] vanishing:", {k: round(v, 3) for k, v in ts.get("vanishing", {}).items()})
txt = rep["text_summary"]
assert "[逐工况误差]" in txt and "[目标值域语义]" in txt, "text_summary 新块缺失"
assert "WORST CASE" in txt or max(p["nmse"] for p in pc) <= 0.3, "A3 worst-case 行缺失"
print("[A3] worst-case 行 OK")

# 2) B3: apply_mutations 禁用算子硬拦截
mut = [{"type": "replace_subtree", "path": [], "new_dsl": "C1*zeta2v - C2*zeta2e"},
       {"type": "replace_subtree", "path": [], "new_dsl": "C1/(zeta2v + K1)"},
       {"type": "perturb_param", "param_id": "C1", "magnitude": 0.1}]
rng = np.random.default_rng(0)
g2 = g.clone()
g2 = llm_ops.apply_mutations(g2, mut, rng, forbidden_ops=["sub", "div"])
ops_after = {n.op for _, n in walk(g2.root)}
assert "sub" not in ops_after and "div" not in ops_after, \
    f"禁用算子泄漏: {ops_after}"
print("[B3] forbidden_ops 拦截 OK (ops:", sorted(ops_after), ")")

# 3) M_GUIDE prompt 硬规则渲染 (heuristic 客户端, 只验证 prompt 组装)
rep["forbidden_ops"] = ["sub", "div"]
import inspect
src = inspect.getsource(llm_ops.M_GUIDE)
assert "HARD RULES" in src and "forbidden_ops" in src, "M_GUIDE 未接硬规则"
# 直接调 M_GUIDE (heuristic 后端会路由走 _mutations_json, 但 prompt 已组装)
muts = llm_ops.M_GUIDE(llm_ops.HeuristicLLM(), [(g, 0.5)], rep)
print("[M_GUIDE] prompt 组装 OK, mutations:", len(muts))

# 4) B2: _inject_wall_seeds
from lgf.pipeline_lgf import _inject_wall_seeds
from lgf.config import LGFConfig
cfg = LGFConfig(population=10)
pop0 = [lgfdsl.parse_dsl("C1*eta2^2", var_names) for _ in range(10)]
pop1 = _inject_wall_seeds(pop0, cs, var_names, cfg)
n_wall = sum(1 for gg in pop1
             if any(n.op == "var" and gg.var_names[n.attrs["index"]] == "yw"
                    for _, n in walk(gg.root)))
assert n_wall >= 3, f"种子注入后含 yw 个体 {n_wall} < 3"
print(f"[B2] wall-seed OK: 10 个无 yw 个体 -> 注入后 {n_wall}/{len(pop1)} 含 yw")

print("ALL_P1D_CHECKS_PASSED")
