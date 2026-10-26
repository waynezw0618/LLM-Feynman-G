# -*- coding: utf-8 -*-
"""v3 预检: 用真实 MiniMax-M2.7 API 跑一个玩具 case, 确认 G_INIT + M_GUIDE
两条通道都真实执行 (llm_stats 可见), 避免烧 3 小时跑批后才发现通道又哑火.

通过判据: ginit.n_parsed > 0 且 mguide.n_calls >= 1 (n_nonempty/applied 尽量 >0).
"""
import os
import re
import sys

import numpy as np

REPO = r"d:\AI\LLM-Feynman-G"
sys.path.insert(0, REPO)

CFG = r"C:\Users\lenovo\.config\evoscientist\config.yaml"
key = None
with open(CFG, encoding="utf-8") as f:
    m = re.search(r"^minimax_api_key:\s*'([^']*)'\s*$|^minimax_api_key:\s*([^\s']\S*)\s*$",
                  f.read(), re.M)
if m:
    key = (m.group(1) or m.group(2) or "").strip().strip("'\"")
if not key:
    sys.exit("[preflight] minimax_api_key not found")

os.environ["LGF_LLM_API_KEY"] = key
os.environ["LGF_LLM_DEBUG"] = "1"

from lgf.config import LGFConfig          # noqa: E402
from lgf import llm_ops                   # noqa: E402
from lgf.pipeline_lgf import run_lgf      # noqa: E402

rng = np.random.default_rng(0)
N = 100
X = rng.uniform(0.1, 2.0, (N, 4))
y = 0.7 * X[:, 0] ** 1.5 * X[:, 1] ** -0.5 + 0.05 * rng.normal(size=N)
cases = [dict(X=X, y=y, name="toy")]
VARS = ["eta2", "zeta2e", "s2o2v", "yw"]

client = llm_ops.make_client("openai", url="https://api.minimaxi.com/v1",
                             model="MiniMax-M2.7", api_key=key, timeout=240.0)
cfg = LGFConfig(population=10, generations=6, llm_every=5, sub_sample=100,
                use_priors=False, time_budget_s=240, seed=1, llm_backend="openai",
                llm_url="https://api.minimaxi.com/v1", llm_model="MiniMax-M2.7",
                llm_api_key=key, llm_timeout=240.0)
res = run_lgf(cases, VARS, cfg, os.path.join(REPO, "outputs", "asm", "_smoke_v3_live"),
              client, name="preflight_mm27")

st = res["llm_stats"]
print("=== preflight llm_stats ===")
print("ginit:", st["ginit"])
print("mguide:", st["mguide"])
print("cfg llm_api_key in final.json:", res["cfg"].get("llm_api_key"))
ok = (st["ginit"]["n_parsed"] > 0 and st["mguide"]["n_calls"] >= 1)
print("PREFLIGHT", "PASS" if ok else "FAIL")
