import json
s = json.load(open(r'd:\AI\LLM-Feynman-G\outputs\asm\runs_heuristic_pope_gsrt\summary_B.json', encoding='utf-8'))
print('targets:', list(s['targets'].keys()))
for k, v in s['targets'].items():
    print(f"{k}: loss={v['loss']:.4f} cplx={v['complexity']['nodes']} :: {v['dsl'][:95]}")
