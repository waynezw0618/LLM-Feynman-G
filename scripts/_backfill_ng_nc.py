# -*- coding: utf-8 -*-
import json, os, sys
sys.path.insert(0, r'd:\AI\LLM-Feynman-G')
from lgf import dsl as lgfdsl
import numpy as np

ds = np.load(r'd:\AI\LLM-Feynman-G\outputs\asm\dataset.npz')
featB = [str(x) for x in ds['featB_keys']]
featA = [str(x) for x in ds['featA_keys']]
RO = [7.5, 15.0, 30.0, 75.0, 150.0]

for sc in ['A', 'B']:
    feat = featB if sc == 'B' else featA
    p = rf'd:\AI\LLM-Feynman-G\outputs\asm\runs_heuristic\summary_{sc}.json'
    if not os.path.exists(p):
        continue
    s = json.load(open(p, encoding='utf-8'))
    s.setdefault('n_cases', len(RO))
    for t, info in s['targets'].items():
        try:
            g = lgfdsl.parse_dsl(info['dsl'], feat)
            info['n_g'] = len(g.global_names())
            info['n_c'] = len(g.case_names())
        except Exception as e:
            info['n_g'] = 0; info['n_c'] = 0; info['decode_err'] = str(e)
    json.dump(s, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(sc, 'n_cases', s['n_cases'])
    for t, i in s['targets'].items():
        print(' ', t, 'n_g', i.get('n_g'), 'n_c', i.get('n_c'),
              'len(theta)', len(i['theta']),
              'dec_ok' if 'decode_err' not in i else 'DEC_ERR')