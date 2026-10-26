import json
def cmp(t):
    h = json.load(open(f"D:/AI/LLM-Feynman-G/outputs/asm/runs_heuristic_gsonly/B_a_{t}_final.json"))["best"]
    m = json.load(open(f"D:/AI/LLM-Feynman-G/outputs/asm/runs_minimax_pope_gsonly/B_a_{t}_final.json"))["best"]
    print(f"=== {t} ===")
    print(f"  heur loss={h['loss']:.4f}  dsl={h['dsl'][:120]}")
    print(f"  minx loss={m['loss']:.4f}  dsl={m['dsl'][:120]}")
    print(f"  cplx heur={h['complexity']['nodes']}n/{h['complexity']['params']}p  vs  minx={m['complexity']['nodes']}n/{m['complexity']['params']}p")
    print(f"  has_yw: heur={'yw' in h['dsl']}  minx={'yw' in m['dsl']}")
    print(f"  5Ro NMSE:")
    for ro in ["Ro7.5","Ro15","Ro30","Ro75","Ro150"]:
        bm = h["metrics"].get(ro + "_nmse", -1)
        mm = m["metrics"].get(ro + "_nmse", -1)
        d = mm - bm
        s = "+" if d >= 0 else ""
        print(f"    {ro:<6} heur={bm:.4f}  minx={mm:.4f}  d={s}{d:.4f}")
    print()
    return h, m

cmp("T4v")
