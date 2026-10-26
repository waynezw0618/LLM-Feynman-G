# -*- coding: utf-8 -*-
"""_probe_trueasm_ginit.py — 真 ASM G_INIT 活性探针 (单次 M2.7 调用)."""
import os
import re
import sys

REPO = r"d:\AI\LLM-Feynman-G"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

CFG = r"C:\Users\lenovo\.config\evoscientist\config.yaml"
key = None
with open(CFG, encoding="utf-8") as f:
    m = re.search(r"^minimax_api_key:\s*'([^']*)'\s*$|^minimax_api_key:\s*([^\s']\S*)\s*$",
                  f.read(), re.M)
if m:
    key = (m.group(1) or m.group(2) or "").strip().strip("'\"")
if not key:
    sys.exit("no key")

from lgf import llm_ops
from rc_asm_true_lgf import PHYSICS_CONTEXT
from rc_asm_true_eval import load_variant
from rc_asm_true_prep import FEAT_NAMES

data = load_variant("v")
cases = [dict(X=c["Xfit"], y=c["yfit"], name=f"Ro{c['ro']:g}") for c in data]
client = llm_ops.make_client("openai", url="https://api.minimaxi.com/v1",
                             model="MiniMax-M2.7", api_key=key, timeout=240.0)

from lgf.priors import data_profile
profile = data_profile(cases, list(FEAT_NAMES))
dsls = llm_ops.G_INIT(client, profile, ["lin", "mul2", "sat"], n=6,
                      use_fewshot=True, extra_context=PHYSICS_CONTEXT)
print(f"n_candidates={len(dsls)}")
n_ok = n_k = 0
for d in dsls:
    g = llm_ops.parse_dsl_safely(d, list(FEAT_NAMES))
    if g is None:
        print(f"  [parse-fail] {d[:100]}")
        continue
    n_ok += 1
    ks = g.case_names()
    if ks:
        n_k += 1
        from lgf.pipeline_lgf import _force_global_params
        _force_global_params(g)
        print(f"  [K→C {ks}] {d[:110]}")
        print(f"    after: {g.to_dsl()[:110]} case_names={g.case_names()}")
    else:
        print(f"  [ok] {d[:110]}")
print(f"parsed {n_ok}/{len(dsls)}, with_K {n_k}")
