# -*- coding: utf-8 -*-
"""rc_asm_true_mix_report.py — MIX 跑批结果汇总 + 假设验证.

输入:
  outputs/asm/runs_trueasm_mix_pysr/result.json
  outputs/asm/runs_trueasm_mix_lgf_go/result.json
  outputs/asm/runs_trueasm_lgf/{v,g,e}_{mg,go}/result.json  (6 LGF)
  outputs/asm/runs_trueasm_pysr/{v,g,e}/result.json          (3 PySR)
输出:
  outputs/asm/ASM_trueasm_mix_report.md   (markdown 报告)
  outputs/asm/compare_with_mix.json       (run 字典)
  outputs/asm/figs_trueasm/fig_mix_compare.png (loss vs rho 散点)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
OUT_DIR = REPO / 'outputs' / 'asm'
FIG_DIR = OUT_DIR / 'figs_trueasm'


def _scalar(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_runs():
    """11 个 run → dict. 9 原始 + mix_pysr + mix_lgf_go."""
    runs = {}
    for v in ('v', 'g', 'e'):
        for arm in ('mg', 'go'):
            p = OUT_DIR / 'runs_trueasm_lgf' / f'{v}_{arm}' / 'result.json'
            if p.exists():
                j = json.load(open(p, encoding='utf-8'))
                b = j['entries'][0]
                m = b.get('metrics', {}) or {}
                runs[f'lgf_{v}_{arm}'] = dict(
                    loss=_scalar(m.get('loss_asm', float('inf'))),
                    rho=_scalar(m.get('rho_a', float('nan'))),
                    dsl=b.get('dsl', ''),
                    complexity=_scalar(b.get('complexity', 0)),
                    per_case=m.get('per_case', {}))
        p = OUT_DIR / 'runs_trueasm_pysr' / v / 'result.json'
        if p.exists():
            j = json.load(open(p, encoding='utf-8'))
            b = j.get('best', {}) or {}
            m = b.get('metrics', {}) or {}
            runs[f'pysr_{v}'] = dict(
                loss=_scalar(m.get('loss_asm', float('inf'))),
                rho=_scalar(m.get('rho_a', float('nan'))),
                dsl=b.get('equation', ''),
                complexity=_scalar(b.get('complexity', 0)),
                per_case=m.get('per_case', {}))
    for k, sub in [('mix_pysr', 'runs_trueasm_mix_pysr'),
                   ('mix_lgf_go', 'runs_trueasm_mix_lgf_go')]:
        p = OUT_DIR / sub / 'result.json'
        if p.exists():
            j = json.load(open(p, encoding='utf-8'))
            if 'best' in j and 'pareto' in j:
                m = j['best']['metrics']
                runs[k] = dict(
                    loss=_scalar(m.get('loss_asm', float('inf'))),
                    rho=_scalar(m.get('rho_a', float('nan'))),
                    dsl=j['best'].get('sympy', j['best'].get('equation', '')),
                    complexity=_scalar(j['best'].get('complexity', 0)),
                    per_case=m.get('per_case', {}),
                    feat_names=j.get('feat_names', []),
                    elapsed_s=j.get('elapsed_s', 0))
            elif 'entries' in j:
                b = j['entries'][0]
                m = b.get('metrics', {}) or {}
                runs[k] = dict(
                    loss=_scalar(m.get('loss_asm', float('inf'))),
                    rho=_scalar(m.get('rho_a', float('nan'))),
                    dsl=b.get('dsl', ''),
                    complexity=_scalar(b.get('complexity', 0)),
                    per_case=m.get('per_case', {}),
                    feat_names=j.get('feat_names', []),
                    elapsed_s=j.get('elapsed_s', 0))
    return runs


def feature_usage(runs, prefix='mix_'):
    """统计 mix run 的特征使用频率 (按 v/g/e 分组)."""
    out = {}
    for k, r in runs.items():
        if not k.startswith(prefix):
            continue
        eq = r['dsl']
        # 解析每个 T#_v / T#_g / T#_e 出现次数
        v_count = sum(eq.count(f'_{vv}') for vv in ('v',) for f in
                      [f'T{m}_v' for m in range(1, 11)] + [f'eta2_v', f'zeta2_v'])
        g_count = sum(eq.count(f'_{vv}') for vv in ('g',) for f in
                      [f'T{m}_g' for m in range(1, 11)] + [f'eta2_g', f'zeta2_g'])
        e_count = sum(eq.count(f'_{vv}') for vv in ('e',) for f in
                      [f'T{m}_e' for m in range(1, 11)] + [f'eta2_e', f'zeta2_e'])
        shared_count = sum(eq.count(f) for f in ['T1', 'T3', 'ret', 'roloc'])
        out[k] = dict(v=v_count, g=g_count, e=e_count, shared=shared_count,
                      total=len(eq))
    return out


def make_loss_rho_scatter(runs, fig_path):
    """loss vs ρ 散点, 11 run, mix 高亮."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'v': '#1f77b4', 'g': '#2ca02c', 'e': '#d62728'}
    markers = {'lgf': 'o', 'pysr': 's'}
    for k, r in runs.items():
        x = r['loss']; y = r['rho']
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if k.startswith('mix_'):
            ax.scatter(x, y, marker='*', s=320, c='gold', edgecolor='black',
                       linewidth=1.4, zorder=10, label=k)
            ax.annotate(k, (x, y), xytext=(8, 8), textcoords='offset points',
                        fontsize=10, weight='bold')
            continue
        # parse: lgf_v_go → color v, marker lgf
        if k.startswith('lgf_'):
            vv = k.split('_')[1]
            mk = 'o'
        elif k.startswith('pysr_'):
            vv = k.split('_')[1]
            mk = 's'
        else:
            continue
        ax.scatter(x, y, c=colors[vv], marker=mk, s=70, alpha=0.8, edgecolor='black',
                   linewidth=0.5)
        ax.annotate(k, (x, y), xytext=(4, 4), textcoords='offset points',
                    fontsize=7, alpha=0.75)
    ax.set_xlabel('loss_asm (lower = better)')
    ax.set_ylabel(r'$\rho_a$ (higher = better)')
    ax.set_title('True-ASM 9 original runs + MIX: loss-$\\rho_a$ tradeoff')
    ax.grid(True, alpha=0.3)
    # legend for variants
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['v'],
                  markersize=10, label='v (W̄)'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['g'],
                  markersize=10, label='g (W̄+Ω)'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['e'],
                  markersize=10, label='e (W̄+2Ω)'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=10, label='● = LGF'),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                  markersize=10, label='■ = PySR'),
           Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                  markeredgecolor='black', markersize=15, label='★ = MIX')]
    ax.legend(handles=leg, loc='lower right', fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    print(f'fig -> {fig_path}')


def make_per_case_rho(runs, fig_path):
    """MIX 与 v_pysr / g_pysr / e_pysr 逐工况 ρ_a 柱状图."""
    cases = ['Ro7.5', 'Ro15', 'Ro30', 'Ro75', 'Ro150']
    keys = ['pysr_v', 'pysr_g', 'pysr_e', 'mix_pysr', 'mix_lgf_go']
    x = np.arange(len(cases))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, k in enumerate(keys):
        r = runs.get(k, {})
        pc = r.get('per_case', {}) or {}
        rhos = [pc.get(c, {}).get('rho_a', np.nan) for c in cases]
        ax.bar(x + (i - 2) * width, rhos, width, label=k,
               edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel(r'$\rho_{a,case}$')
    ax.set_title('Per-case $\\rho_a$ — MIX vs single-variant baselines')
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    print(f'fig -> {fig_path}')


def main():
    runs = load_runs()
    print(f'Loaded {len(runs)} runs: {list(runs.keys())}')
    # save runs json
    with open(OUT_DIR / 'compare_with_mix.json', 'w', encoding='utf-8') as f:
        json.dump(runs, f, ensure_ascii=False, indent=2, default=str)

    # table
    print(f"\n{'run':<22} {'loss_asm':>10} {'rho_a':>8} {'cplx':>5}  dsl/equation (truncated)")
    for k in ['lgf_v_go', 'lgf_g_go', 'lgf_e_go', 'lgf_v_mg', 'lgf_g_mg', 'lgf_e_mg',
              'pysr_v', 'pysr_g', 'pysr_e', 'mix_lgf_go', 'mix_pysr']:
        if k not in runs:
            continue
        r = runs[k]
        dsl = (r.get('dsl', '') or '')[:60].replace('\n', ' ')
        print(f'{k:<22} {r["loss"]:>10.4f} {r["rho"]:>+8.4f} {r["complexity"]:>5.0f}  {dsl}')

    # feature usage
    usage = feature_usage(runs)
    if usage:
        print('\nMIX 公式中 v/g/e 通道使用计数:')
        for k, u in usage.items():
            print(f'  {k}: v={u["v"]}  g={u["g"]}  e={u["e"]}  shared(T1/T3/ret/roloc)={u["shared"]}')

    # figs
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_loss_rho_scatter(runs, FIG_DIR / 'fig_mix_compare.png')
    make_per_case_rho(runs, FIG_DIR / 'fig_mix_percase_rho.png')


if __name__ == '__main__':
    main()
