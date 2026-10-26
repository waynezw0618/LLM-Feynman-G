# -*- coding: utf-8 -*-
"""探针: 直接调 G_INIT, 打印 MiniMax 原始候选与逐条 parse 失败原因."""
import os
import re
import sys

sys.path.insert(0, r"d:\AI\LLM-Feynman-G")

CFG = r"C:\Users\lenovo\.config\evoscientist\config.yaml"
key = None
with open(CFG, encoding="utf-8") as f:
    m = re.search(r"^minimax_api_key:\s*'([^']*)'\s*$|^minimax_api_key:\s*([^\s']\S*)\s*$",
                  f.read(), re.M)
if m:
    key = (m.group(1) or m.group(2) or "").strip().strip("'\"")
os.environ["LGF_LLM_API_KEY"] = key

import numpy as np                        # noqa: E402
from lgf import llm_ops                  # noqa: E402
from lgf import dsl as lgfdsl            # noqa: E402
from lgf.templates import TEMPLATE_REGISTRY  # noqa: E402
from lgf.pipeline_lgf import _quick_profile  # noqa: E402

rng = np.random.default_rng(0)
N = 100
X = rng.uniform(0.1, 2.0, (N, 4))
y = 0.7 * X[:, 0] ** 1.5 * X[:, 1] ** -0.5 + 0.05 * rng.normal(size=N)
cases = [dict(X=X, y=y, name="toy")]
VARS = ["eta2", "zeta2e", "s2o2v", "yw"]
profile = _quick_profile(cases, VARS)

client = llm_ops.make_client("openai", url="https://api.minimaxi.com/v1",
                             model="MiniMax-M2.7", api_key=key, timeout=240.0)
dsls = llm_ops.G_INIT(client, profile, list(TEMPLATE_REGISTRY.keys()), n=8)
print("n dsls returned:", len(dsls))
for d in dsls:
    try:
        g = lgfdsl.parse_dsl(d, VARS)
        print("  OK  :", d[:90])
    except Exception as e:
        print("  FAIL:", repr(d[:90]), "->", type(e).__name__, str(e)[:80])
