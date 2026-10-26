"""V3 冒烟: ① M_GUIDE 修复后是否真正执行 (n_calls/n_applied/n_alive); ②
forbidden_ops eval 层拦截是否生效 (forbidden_filtered>0 且 best 无 sub/div).
"""
import sys

import numpy as np

sys.path.insert(0, r"d:\AI\LLM-Feynman-G")
from lgf.config import LGFConfig
from lgf import llm_ops
from lgf.pipeline_lgf import run_lgf

rng = np.random.default_rng(0)
N = 120
X = rng.uniform(0.1, 2.0, (N, 4))
y = 0.7 * X[:, 0] ** 1.5 * X[:, 1] ** -0.5 + 0.05 * rng.normal(size=N)
cases = [dict(X=X, y=y, name="toy")]
VARS = ["eta2", "zeta2e", "s2o2v", "yw"]
client = llm_ops.make_client("heuristic")


def go(tag, **kw):
    cfg = LGFConfig(population=12, generations=6, llm_every=5, sub_sample=100,
                    use_priors=False, time_budget_s=90, seed=1,
                    llm_backend="heuristic", **kw)
    res = run_lgf(cases, VARS, cfg, rf"d:\AI\LLM-Feynman-G\outputs\asm\_smoke_v3",
                  client, name=tag)
    st = res["llm_stats"]
    print(f"--- {tag} ---")
    print(f"  mguide: {st['mguide']}")
    print(f"  forbidden_filtered={st['forbidden_filtered']}  ginit={st['ginit']}")
    for r in res["log"]:
        print(f"  gen={r['gen']} n_pop={r['n_pop']} best={r['best_loss']:.4g}")
    print(f"  best dsl: {res['best']['dsl'][:100]}")
    return res


go("smoke_v3a")
go("smoke_v3b", forbidden_ops=["sub", "div"])
