import json
print('=== 关键 best 公式复检 ===')
items = [
    ('mm_bare_mg T2v 0.9625',  r'outputs\asm\runs_v3_mm_bare_mg\B_a_T2v_final.json'),
    ('mm_bare_go T5g 0.0000',  r'outputs\asm\runs_v3_mm_bare_go\B_a_T5g_final.json'),
    ('mm_bare_mg T3v 0.4065',  r'outputs\asm\runs_v3_mm_bare_mg\B_a_T3v_final.json'),
    ('heur_bare_mg T3v 0.6833',r'outputs\asm\runs_v3_heur_bare_mg\B_a_T3v_final.json'),
    ('heur_bare_go T3v 0.7968',r'outputs\asm\runs_v3_heur_bare_go\B_a_T3v_final.json'),
    ('heur_bare_mg T5g 0.6110',r'outputs\asm\runs_v3_heur_bare_mg\B_a_T5g_final.json'),
    ('mm_full_go T5g 0.7447',  r'outputs\asm\runs_v3_mm_full_go\B_a_T5g_final.json'),
    ('mm_full_mg T2v 0.0579',  r'outputs\asm\runs_v3_mm_full_mg\B_a_T2v_final.json'),
    ('heur_full_mg T2v 0.0579',r'outputs\asm\runs_v3_heur_full_mg\B_a_T2v_final.json'),
]
for label, p in items:
    try:
        d = json.load(open(p, encoding='utf-8'))
        b = d['best']
        c = b.get('complexity', {})
        theta = b.get('theta', [])
        theta_nz = [(i, round(t, 2)) for i, t in enumerate(theta) if abs(t) > 1]
        m = b.get('metrics', {})
        rmse_v = [v for k, v in m.items() if k.endswith('_rmse')]
        nmse_v = [v for k, v in m.items() if k.endswith('_nmse')]
        print('--', label)
        print('   dsl :', b['dsl'])
        print('   theta (|.|>1):', theta_nz)
        print('   max|theta|=', round(max(abs(t) for t in theta), 2),
              ' nodes=', c.get('nodes', '?'))
        if rmse_v:
            print(f'   RMSE max/mean: {max(rmse_v):.1f}/{sum(rmse_v)/len(rmse_v):.1f}'
                  f'   NMSE mean: {sum(nmse_v)/len(nmse_v):.3f}')
    except Exception as e:
        print('--', label, ': ERR', e)
