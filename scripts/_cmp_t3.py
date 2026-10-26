import json
h = json.load(open(r"D:/AI/LLM-Feynman-G/outputs/asm/runs_heuristic_gsonly/B_a_T3v_final.json"))["best"]
m = json.load(open(r"D:/AI/LLM-Feynman-G/outputs/asm/runs_minimax_pope_gsonly/B_a_T3v_final.json"))["best"]
print(f"heur T3v: loss={h['loss']:.4f}  dsl={h['dsl'][:130]}")
print(f"minx T3v: loss={m['loss']:.4f}  dsl={m['dsl'][:130]}")
print()
print(f"heur cplx: nodes={h['complexity']['nodes']} params={h['complexity']['params']}")
print(f"minx cplx: nodes={m['complexity']['nodes']} params={m['complexity']['params']}")
print()
print("T3v 5 Ro NMSE 对比")
print(f"  {'Ro':<6}  {'heur':>8}  {'minx':>8}  {'delta':>9}")
for ro in ["Ro7.5","Ro15","Ro30","Ro75","Ro150"]:
    bm = h["metrics"].get(ro + "_nmse", -1)
    mm = m["metrics"].get(ro + "_nmse", -1)
    d = mm - bm
    s = "+" if d >= 0 else ""
    print(f"  {ro:<6}  {bm:>8.4f}  {mm:>8.4f}  {s}{d:>8.4f}")
