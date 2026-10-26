"""rc_asm_run.py — 阶段2: LGF 符号回归 (方案B: a_m(不变量); 方案A: b_ij 直接回归).

用法:
  python scripts/rc_asm_run.py --scheme B --backend heuristic
  python scripts/rc_asm_run.py --scheme A --backend heuristic
  python scripts/rc_asm_run.py --scheme B --backend glm       # 需 LGF_LLM_API_KEY
  python scripts/rc_asm_run.py --scheme B --backend deepseek --llm_model deepseek-chat
  python scripts/rc_asm_run.py --scheme B --backend minimax
输出: outputs/asm/runs_{tag}/{scheme}_{target}_final.json + summary_{scheme}.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lgf.config import LGFConfig
from lgf import llm_ops
from lgf import dsl as lgfdsl
from lgf.pipeline_lgf import run_lgf

OUT_DIR = os.path.join(REPO, "outputs", "asm")
RO_CASES = [7.5, 15.0, 30.0, 75.0, 150.0]
COMP6 = ["uu", "vv", "ww", "uv", "uw", "vw"]

BACKENDS = {
    "heuristic": dict(backend="heuristic", url=None, model=None),
    "glm": dict(backend="openai", url="https://open.bigmodel.cn/api/paas/v4", model="glm-5.1"),
    # V4 (2026-10-12): GLM 5.3 思考模型 — 强制思考且 reasoning_effort 默认 max,
    # 必须显式降 low + max_tokens 16384 (否则 reasoning 吃光配额 content 为空;
    # 实测 low 档 G_INIT 30 候选 13s, 解析率 29/30).
    "glm53": dict(backend="openai", url="https://open.bigmodel.cn/api/paas/v4",
                  model="glm-5.3", max_tokens=16384,
                  extra_body={"thinking": {"type": "enabled"},
                              "reasoning_effort": "low"}),
    "deepseek": dict(backend="openai", url="https://api.deepseek.com/v1", model="deepseek-chat"),
    "minimax": dict(backend="openai", url="https://api.minimaxi.com/v1", model="MiniMax-M2.7"),
}


def build_cases(ds, scheme, ycol, sel_names, fitness="coef", fit_mask="valid_fit",
                wall_boost=1.0, wall_region=0.7, comp_w=None):
    """构造 LGF cases.

    P0-C 加权 (2026-09-30):
    - fit_mask: "valid_fit" → 用双掩码中干净的反演目标训练 (cond/amax 剔除,
      见 rc_pope_prep.py v2); "valid" → 全物理有效点 (旧行为).
    - wall_boost / wall_region: |y/h|>wall_region 的点权重 ×wall_boost (近壁加权,
      对症近壁失配; fitness=coef 与 stress 均生效).
    - comp_w (dict, 分量→权重): stress fitness 权重从 ||T_m||²_F 推广为
      Σ_c α_c·T_m[c]², 对症 vw/uw 支撑被对角分量淹没 (RC3).
    """
    cs = []
    comp_w = comp_w or {}
    for i in range(len(RO_CASES)):
        p = f"c{i}_"
        vkey = fit_mask if (fit_mask != "valid" and (p + fit_mask) in ds.files) else "valid"
        v = ds[p + vkey]
        X = ds[p + ("featB" if scheme == "B" else "featA")][v]
        if scheme == "B":
            col = sel_names.index(ycol)
        else:
            col = COMP6.index(ycol)
        y = ds[p + "a"][:, col][v] if scheme == "B" else ds[p + "b6"][:, col][v]
        c = dict(X=X, y=y, name=f"Ro{RO_CASES[i]:g}")
        if (p + "yh") in ds.files:
            yh = np.abs(ds[p + "yh"][v])
            wall_mult = 1.0 + (wall_boost - 1.0) * (yh > wall_region).astype(float)
        else:
            wall_mult = np.ones(len(y))
        if scheme == "B" and fitness == "stress":
            # 分量加权应力空间权重: w = Σ_c α_c·T_m[c]²  (α≡1 时退化为 ||T_m||²_F)
            Tm = ds[p + "T_" + ycol][v]          # (Nv, 6) 基张量六分量
            if any(abs(comp_w.get(cc, 1.0) - 1.0) > 1e-12 for cc in COMP6):
                w = np.zeros(len(y))
                for kk, cc in enumerate(COMP6):
                    w += comp_w.get(cc, 1.0) * Tm[:, kk] ** 2
            else:
                w = np.sum(Tm ** 2, axis=1)
            c["w"] = w * wall_mult
        elif wall_boost != 1.0:
            c["w"] = wall_mult                    # coef fitness 也支持近壁加权
        cs.append(c)
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["A", "B"], required=True)
    ap.add_argument("--backend", choices=list(BACKENDS), default="heuristic")
    ap.add_argument("--tag", default=None, help="输出目录标签 (默认=backend)")
    ap.add_argument("--budget", type=float, default=120.0, help="每目标时间预算(s)")
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--llm_api_key", default=None)
    ap.add_argument("--dataset", default="dataset.npz",
                    help="数据集文件名 (相对 outputs/asm, 如 dataset_vort.npz)")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="B: S SOm S2 Om2 Om2S; A: uu vv ww uv uw vw")
    ap.add_argument("--fitness", choices=["coef", "stress"], default="coef",
                    help="coef=系数空间 NMSE (默认); stress=应力空间加权 NMSE "
                         "(需数据集含 c{i}_T_{target} 基张量列, 如 dataset_pope*.npz)")
    ap.add_argument("--fit_mask", choices=["valid", "valid_fit"], default="valid_fit",
                    help="训练掩码: valid_fit=剔除病态反演点 (cond/amax, P0-B); "
                         "valid=全物理有效点 (旧行为, 无 valid_fit 键时自动回退)")
    ap.add_argument("--wall_boost", type=float, default=1.0,
                    help="近壁加权: |y/h|>wall_region 的点权重倍数 (P0-C, 1=关闭)")
    ap.add_argument("--wall_region", type=float, default=0.7,
                    help="近壁区域阈值 |y/h|>")
    ap.add_argument("--w_uu", type=float, default=1.0)
    ap.add_argument("--w_vv", type=float, default=1.0)
    ap.add_argument("--w_ww", type=float, default=1.0)
    ap.add_argument("--w_uv", type=float, default=1.0)
    ap.add_argument("--w_uw", type=float, default=1.0)
    ap.add_argument("--w_vw", type=float, default=1.0)
    ap.add_argument("--forbid_map", default="",
                    help="P1d-B3 目标级禁用算子映射, 形如 'T5g:sub,div;T1:div' "
                         "(分号分隔多个目标)")
    ap.add_argument("--ginit_only", action="store_true",
                    help="P1d-C2: LLM 仅出 G_INIT 初始种群, M_GUIDE 全程关闭")
    ap.add_argument("--wall_seeds", type=int, default=1, choices=[0, 1],
                    help="P1d-B2 壁面特征程序化种子注入 (1=开, 默认; 0=关, 裸配置消融)")
    # ===== v3a: G_INIT 鲁棒性 (A1 解析重试 + A3 few-shot) =====
    ap.add_argument("--ginit_retry", type=int, default=2,
                    help="G_INIT 解析率不达标时最大重试次数 (0=单次, 2=默认最多 3 次)")
    ap.add_argument("--ginit_min_kept", type=int, default=3,
                    help="G_INIT 累积解析成功数 >= 此值停止重试 (默认 3)")
    ap.add_argument("--ginit_fewshot", type=int, default=1, choices=[0, 1],
                    help="G_INIT prompt 注入 few-shot 合法 DSL 示例 (1=开, 默认; 0=关)")
    args = ap.parse_args()

    forbid_map: dict = {}
    for part in (args.forbid_map or "").split(";"):
        if not part.strip():
            continue
        k, _, v = part.partition(":")
        ops = [x.strip() for x in v.split(",") if x.strip()]
        if k.strip() and ops:
            forbid_map[k.strip()] = ops

    ds = np.load(os.path.join(OUT_DIR, args.dataset))
    tag = args.tag or args.backend
    log_dir = os.path.join(OUT_DIR, f"runs_{tag}")
    os.makedirs(log_dir, exist_ok=True)

    be = BACKENDS[args.backend]
    model = args.llm_model or be["model"]
    api_key = (args.llm_api_key or os.environ.get("LGF_LLM_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    client = llm_ops.make_client(be["backend"], url=be["url"], model=model,
                                 api_key=api_key, timeout=300.0,
                                 max_tokens=be.get("max_tokens"),
                                 extra_body=be.get("extra_body"))
    if args.backend != "heuristic" and not api_key:
        print(f"[run][WARN] 后端 {args.backend} 无 API key, LGF 会静默退化启发式, "
              f"结果不可计入 LLM 后端对比!")

    sel_names = [str(x) for x in ds["sel_basis"]]
    var_names = [str(x) for x in (ds["featB_keys"] if args.scheme == "B"
                                  else ds["featA_keys"])]
    targets = args.targets or (sel_names if args.scheme == "B" else COMP6)

    summary = dict(scheme=args.scheme, backend=args.backend, tag=tag, model=model,
                   var_names=var_names, budget=args.budget, n_cases=len(RO_CASES),
                   fitness=args.fitness, dataset=args.dataset,
                   fit_mask=args.fit_mask, wall_boost=args.wall_boost,
                   wall_region=args.wall_region,
                   comp_w={c: getattr(args, "w_" + c) for c in COMP6},
                   forbid_map=forbid_map, ginit_only=bool(args.ginit_only),
                   wall_seeds=bool(args.wall_seeds),
                   ginit_retry=int(args.ginit_retry),
                   ginit_min_kept=int(args.ginit_min_kept),
                   ginit_fewshot=bool(args.ginit_fewshot),
                   targets={})
    for tgt in targets:
        cs = build_cases(ds, args.scheme, tgt, sel_names, fitness=args.fitness,
                         fit_mask=args.fit_mask, wall_boost=args.wall_boost,
                         wall_region=args.wall_region,
                         comp_w={c: getattr(args, "w_" + c) for c in COMP6})
        seed = zlib.crc32(f"{args.scheme}|{tgt}".encode()) % (2**31)
        cfg = LGFConfig(population=args.pop, generations=args.gens, sub_sample=250,
                        time_budget_s=args.budget, final_starts=8, final_maxiter=120,
                        llm_backend=be["backend"], llm_url=be["url"], llm_model=model,
                        llm_api_key=api_key, llm_timeout=240.0, seed=seed,
                        forbidden_ops=forbid_map.get(tgt, []),
                        llm_ginit_only=bool(args.ginit_only),
                        wall_seeds=bool(args.wall_seeds),
                        ginit_retry=int(args.ginit_retry),
                        ginit_min_kept=int(args.ginit_min_kept),
                        ginit_fewshot=bool(args.ginit_fewshot))
        t0 = time.time()
        res = run_lgf(cs, var_names, cfg, log_dir, client,
                      name=f"{args.scheme}_{'a_' if args.scheme == 'B' else 'b_'}{tgt}")
        best = res["best"]
        # 推断 n_g/n_c: 重编译 DSL
        n_g = n_c = 0
        try:
            g = lgfdsl.parse_dsl(best["dsl"], var_names)
            n_g = len(g.global_names()); n_c = len(g.case_names())
        except Exception:
            pass
        summary["targets"][tgt] = dict(
            dsl=best["dsl"], loss=best["loss"], complexity=best["complexity"],
            theta=best["theta"], metrics=best["metrics"],
            n_g=n_g, n_c=n_c,
            llm_stats=res.get("llm_stats", {}),
            elapsed_s=round(time.time() - t0, 1))
        st = res.get("llm_stats", {}).get("mguide", {})
        print(f"[run] {args.scheme}:{tgt:<4} loss={best['loss']:.6g} "
              f"cplx={best['complexity']} {time.time()-t0:.0f}s :: {best['dsl'][:120]}"
              + (f" || M_GUIDE calls={st.get('n_calls',0)} nonempty={st.get('n_nonempty',0)} "
                 f"applied={st.get('n_applied',0)} alive={st.get('n_alive',0)}"
                 if st else ""))

    out = os.path.join(log_dir, f"summary_{args.scheme}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[run] -> {out}")


if __name__ == "__main__":
    main()
