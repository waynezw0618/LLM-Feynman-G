import glob, json
from collections import defaultdict

def load(p):
    d = json.load(open(p, encoding='utf-8'))
    return {
        'tgt': d['name'].split('_')[-1],
        'loss': d['best']['loss'],
        'comp_n': d['best']['complexity']['nodes'],
        'comp_p': d['best']['complexity']['params'],
        'mg_c': d['llm_stats']['mguide']['n_calls'],
        'mg_a': d['llm_stats']['mguide']['n_applied'],
        'mg_lv': d['llm_stats']['mguide']['n_alive'],
        'ginit_k': d['llm_stats']['ginit'].get('n_parsed', 0),
        'ginit_c': d['llm_stats']['ginit'].get('n_candidates', 0),
    }

arms = ['v3_heur_bare_mg', 'v3_mm_bare_mg', 'v3_heur_bare_go', 'v3_mm_bare_go']
print('='*100)
print('|  臂            | tgt |  loss  |  N  | #p | mg.call | mg.appl | mg.alive | ginit kept/cand |')
print('='*100)
for arm in arms:
    rows = []
    for p in sorted(glob.glob(f'outputs/asm/runs_{arm}/*_final.json')):
        rows.append(load(p))
    if not rows: continue
    print(f'-- {arm} --')
    for r in rows:
        print(f"| {arm:14s} | {r['tgt']:3s} | {r['loss']:.4f} | {r['comp_n']:>3d} | {r['comp_p']:>2d} | {r['mg_c']:>7d} | {r['mg_a']:>7d} | {r['mg_lv']:>8d} | {r['ginit_k']:>2d}/{r['ginit_c']:>2d}        |")
print('='*100)
print()
print('--- 主判据: bare + M_GUIDE (heur vs mm) ---')
def mean(arm, k):
    vals = [load(p)[k] for p in glob.glob(f'outputs/asm/runs_{arm}/*_final.json')]
    return sum(vals)/len(vals) if vals else None
h = mean('v3_heur_bare_mg', 'loss')
m = mean('v3_mm_bare_mg', 'loss')
print(f'  bare_mg mean loss: heur = {h:.4f},  mm = {m:.4f},  delta(mm-heur) = {m-h:+.4f}')
print()
print('--- M_GUIDE 净效应 (mg vs go, 同后端) ---')
hh = mean('v3_heur_bare_mg', 'loss')
hh2 = mean('v3_heur_bare_go', 'loss')
mm = mean('v3_mm_bare_mg', 'loss')
mm2 = mean('v3_mm_bare_go', 'loss')
print(f'  heur: mg={hh:.4f} vs go={hh2:.4f} -> delta(mg-go) = {hh-hh2:+.4f}')
print(f'  mm  : mg={mm:.4f} vs go={mm2:.4f} -> delta(mg-go) = {mm-mm2:+.4f}')
