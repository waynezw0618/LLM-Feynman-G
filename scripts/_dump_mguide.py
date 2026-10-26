"""Dump actual M_GUIDE prompt with pope_y setup to show LLM what it sees."""
import os
import sys
sys.path.insert(0, "d:/AI/LLM-Feynman-G")
from lgf.attribution import FEATURE_ROLES
from lgf.llm_ops import M_GUIDE

class _StubLLM:
    def complete(self, prompt, *, n=1, params=None):
        print("=" * 70)
        print("M_GUIDE PROMPT THAT LLM SEES:")
        print("=" * 70)
        print(prompt)
        print("=" * 70)
        return ['{"mutations":[]}']

stub = _StubLLM()
report = {
    "text_summary": "[误差归因报告]\n  节点 0/0/1: sensitivity=0.412, shape=decreasing, std=2.5",
    "feature_roles": FEATURE_ROLES,
    "ypd_bin_errors": [
        {"bin":"[0,15)","share":0.61,"mean_abs_resid":0.18,"n":1200},
        {"bin":"[15,40)","share":0.22,"mean_abs_resid":0.09,"n":2400},
        {"bin":"[40,80)","share":0.12,"mean_abs_resid":0.05,"n":3600},
        {"bin":"[80,inf)","share":0.05,"mean_abs_resid":0.03,"n":4800},
    ],
}
M_GUIDE(stub, [(None, 0.262)], report)