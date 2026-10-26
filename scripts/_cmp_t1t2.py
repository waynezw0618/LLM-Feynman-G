import json
def L(tag, f, t):
    d = json.load(open(f))
    b = d["best"]
    print(f"--- {tag}: {t} ---")
    print(f"  loss={b['loss']:.4f}  dsl={b['dsl'][:140]}")
    c = b["complexity"]
    print(f"  cplx: nodes={c['nodes']} params={c['params']} depth={c['depth']}")
    has_yw = "yw" in b["dsl"]
    print(f"  has yw: {has_yw}")
    return b

def cmp(t):
    bh = L("heuristic", f"D:/AI/LLM-Feynman-G/outputs/asm/runs_heuristic_gsonly/B_a_{t}_final.json", t)
    mh = L("minimax",   f"D:/AI/LLM-Feynman-G/outputs/asm/runs_minimax_pope_gsonly/B_a_{t}_final.json", t)
    print(f"\n=== {t} 5 Ro NMSE 对比 ===")
    for ro in ["Ro7.5", "Ro15", "Ro30", "Ro75", "Ro150"]:
        bm = bh["metrics"].get(ro + "_nmse")
        mm = mh["metrics"].get(ro + "_nmse")
        if bm is not None and mm is not None:
            d = mm - bm
            s = "+" if d >= 0 else ""
            print(f"  {ro:<6s}  heur={bm:.4f}  minx={mm:.4f}  delta={s}{d:.4f}")
    print()
    return bh, mh

for t in ["T1", "T2v"]:
    cmp(t)
