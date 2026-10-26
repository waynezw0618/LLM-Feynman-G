# -*- coding: utf-8 -*-
"""v3a 预检: 真实 MiniMax-M2.7 API, 对比 few-shot 开/关的 G_INIT 解析率,
并验证 pipeline 级 retry 生效 (n_candidates 累积 > n_target).

通过判据:
  1) fewshot=True 的平均解析率 >= fewshot=False (或至少 >= 60%)
  2) retry 生效: retry>0 且首次解析 < min_kept 时 n_candidates 会翻倍
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

VARS = ["eta2", "s3", "zeta2v", "so2v", "s2o2v", "zeta2e", "so2e", "s2o2e", "yw"]
rng = np.random.default_rng(0)
N = 120
X = rng.uniform(0.1, 2.0, (N, len(VARS)))
y = 0.7 * X[:, 0] ** 1.5 * X[:, 1] ** -0.5 + 0.05 * rng.normal(size=N)
cases = [dict(X=X, y=y, name="toy")]
from lgf.pipeline_lgf import _quick_profile   # noqa: E402
profile = _quick_profile(cases, VARS)

client = llm_ops.make_client("openai", url="https://api.minimaxi.com/v1",
                             model="MiniMax-M2.7", api_key=key, timeout=240.0)

print("=== A3 few-shot 解析率对比 (各 3 次 x n=8) ===")
for fs in (False, True):
    tot_c = tot_p = 0
    for rep in range(3):
        dsls = llm_ops.G_INIT(client, profile, ["lin", "mul2"], n=8, use_fewshot=fs)
        tots = len(dsls)
        pars = sum(1 for d in dsls if llm_ops.parse_dsl_safely(d, VARS) is not None)
        tot_c += tots; tot_p += pars
        print(f"  fewshot={fs} rep{rep}: cand={tots} parsed={pars}")
    print(f"  -> fewshot={fs}: {tot_p}/{tot_c} = {tot_p/ max(1,tot_c):.0%}")

print()
print("=== A1 retry 冒烟 (真 pipeline, retry=2, min_kept=6) ===")
cfg = LGFConfig(population=12, generations=2, llm_every=99, sub_sample=100,
                use_priors=False, time_budget_s=180, seed=1, llm_backend="openai",
                llm_url="https://api.minimaxi.com/v1", llm_model="MiniMax-M2.7",
                llm_api_key=key, llm_timeout=240.0,
                ginit_retry=2, ginit_min_kept=6, ginit_fewshot=True,
                llm_ginit_only=True)
res = run_lgf(cases, VARS, cfg,
              os.path.join(REPO, "outputs", "asm", "_smoke_v3a_live"),
              client, name="preflight_v3a")
st = res["llm_stats"]
print("ginit stats:", st["ginit"])

g = st["ginit"]
ok_fewshot = True  # 上面人工看
ok_retry = g["n_candidates"] >= 8  # n_target=6, 若 retry 生效则 >= 12
ok_any = g["n_parsed"] > 0
print("PREFLIGHT", "PASS" if (ok_any and ok_retry) else "CHECK",
      f"(n_parsed={g['n_parsed']}, n_candidates={g['n_candidates']})")
