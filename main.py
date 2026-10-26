"""LGF 统一命令行入口.

示例:
  # 用启发式 LLM (离线) 跑单个基准
  python main.py --framework lgf --benchmark esat --generations 20 --population 40

  # 跑多个基准
  python main.py --framework lgf --suite quick

  # 从 config.xml 加载 LLM 配置 (url / api_key / model)
  python main.py --framework lgf --benchmark esat --config config.xml

  # 用 mock Flask 服务器 (http://localhost:5050/completions)
  python scripts/mock_llm_server.py --port 5050 &
  python main.py --framework lgf --benchmark esat_mc --llm_backend flask --llm_url http://localhost:5050/completions

  # 用外部 LLM-Feynman 服务 (可选, 仅当启用了对应服务时)
  #   路径参考: D:\\AI\\LLM-Feynman\\LLM-Feynman-main\\llm_feynman\\llm_engine.py
  python main.py --framework lgf --benchmark mul_sep --llm_backend flask --llm_url http://localhost:5000/completions

  # 用 OpenAI 兼容 API (url 只填到版本段, 不带 /chat/completions)
  python main.py --framework lgf --benchmark hollomon --llm_backend openai \
      --llm_url https://api.openai.com/v1 --llm_model gpt-4o-mini

  # 用本地无鉴权服务 (vLLM / Ollama / LM Studio), 无需 api_key
  python main.py --framework lgf --benchmark esat --llm_backend openai \
      --llm_url http://localhost:8000/v1 --llm_model Qwen2.5-7B-Instruct

  # 本地 OpenAI 兼容服务 (推荐走环境变量, 避免 key 落在命令行历史里)

  #   PowerShell:  $env:LGF_LLM_API_KEY = "sk-xxxx"
  #   CMD:         set LGF_LLM_API_KEY=sk-xxxx
  #   (也可以 --llm_api_key sk-xxxx 显式传入, 或 setx 做持久化)

  python main.py --framework lgf --benchmark hollomon --llm_backend openai \
      --llm_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
      --llm_model qwen-plus --llm_temperature 0.3 --llm_timeout 120

  # 排障: 加上 --llm_debug, 后端失败会打印具体原因 (401 / 404 / 超时)
  python main.py --framework lgf --benchmark hollomon --llm_backend openai \
      --llm_url https://api.openai.com/v1 --llm_model gpt-4o-mini --llm_debug

配置优先级 (低 → 高, 高覆盖低):
    1. LGFConfig dataclass 内置默认值
    2. config.xml 文件 (通过 --config 指定, 缺省尝试 ./config.xml)
    3. CLI 参数 (任何显式给的值)

LLM 配置:
    --llm_backend   heuristic(默认,离线) | flask(HTTP /completions) | openai(OpenAI 兼容)
    --llm_url       flask 填到 .../completions; openai 只填到 .../v1
    --llm_model     留空则 flask 忽略 / openai 回退 gpt-3.5-turbo
    --llm_api_key   留空则依次读 LGF_LLM_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY
    提示: 任意外部后端不可达时不会报错, 只会退化为启发式输出; 用 --llm_debug 观察.

独立性:
    本命令运行无需 D:\\AI\\LLM-Feynman 任何文件. 仅在显式启用 flask/openai
    后端时才需要对应服务可达; 离线开发默认使用 --llm_backend heuristic.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["lgf", "llmsr"], default="lgf")
    p.add_argument("--benchmark", type=str, default="esat")
    p.add_argument("--suite", choices=["quick", "all", "none"], default="none")
    p.add_argument("--real", type=str, default="", help="真实数据集名, 如 osc1 / osc2 / bactgrow")
    p.add_argument("--n_samples", type=int, default=400)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_llm", action="store_true")
    p.add_argument("--no_priors", action="store_true")
    p.add_argument("--no_attribution", action="store_true")
    p.add_argument("--config", type=str, default="",
                   help="XML 配置文件路径; 缺省留空, 运行时如存在 ./config.xml 才自动加载. "
                        "CLI 参数永远覆盖文件值.")
    # ----- LLM 后端参数: 默认值改为 None / 空串, 便于分层覆盖 -----
    p.add_argument("--llm_backend", choices=["heuristic", "flask", "openai"], default=None,
                   help="None 时回退到 config.xml 或内置默认 'heuristic'.")
    p.add_argument("--llm_url", type=str, default="",
                   help="flask: http://host:port/completions | openai: http://host:port/v1; "
                        "空串表示沿用 config.xml / None.")
    p.add_argument("--llm_model", type=str, default="",
                   help="模型名; 留空则 flask 忽略 / openai 回退 gpt-3.5-turbo.")
    p.add_argument("--llm_api_key", type=str, default="",
                   help="API key; 留空则按 LGF_LLM_API_KEY > OPENAI_API_KEY > DASHSCOPE_API_KEY 解析.")
    p.add_argument("--llm_temperature", type=float, default=None,
                   help="None 时沿用 config.xml / 内置默认 0.7.")
    p.add_argument("--llm_timeout", type=float, default=None,
                   help="None 时沿用 config.xml / 内置默认 60.0.")
    p.add_argument("--llm_debug", action="store_true",
                   help="打印后端调用失败详情 (默认静默退化为启发式).")
    p.add_argument("--max_edges", type=int, default=14)
    p.add_argument("--max_params", type=int, default=6)
    p.add_argument("--time_budget_s", type=float, default=180.0)
    p.add_argument("--log_dir", type=str, default="outputs/runs")
    p.add_argument("--list_benchmarks", action="store_true")
    return p.parse_args()


def cmd_list_benchmarks():
    from lgf.benchmarks import list_benchmarks
    print("合成基准:", list_benchmarks())
    print("真实数据集 (可选, 需放在 D:\\AI\\LLM-Feynman\\data 或自定义路径): "
          "osc1 / osc2 / bactgrow")


def load_config_layered(args) -> "LGFConfig":
    """三层叠加: dataclass 默认 → config.xml → CLI 显式值.

    任何 CLI 字段仅在用户**显式**传入时覆盖 config.xml; 未传则沿用文件值.
    这样可以让 config.xml 当主配置源, CLI 仅做临时覆盖.
    """
    from lgf.config import LGFConfig
    cfg = LGFConfig()                                       # 1. dataclass 默认

    # 2. config.xml (--config 缺省 "" 时, 如 ./config.xml 存在则加载)
    xml_path = args.config or "config.xml"
    if os.path.exists(xml_path):
        cfg = LGFConfig.from_xml(xml_path)
        print(f"[LGF-Config] 已加载 {xml_path} (llm_backend={cfg.llm_backend}, "
              f"llm_url={cfg.llm_url or '-'}, llm_debug={cfg.llm_debug})")
    elif args.config:
        print(f"[LGF-Config] 警告: --config 指定的 {xml_path} 不存在, 仅用默认 + CLI.")

    # 3. CLI 显式覆盖 (None / "" / False 默认都不覆盖)
    if args.llm_backend is not None:        cfg.llm_backend = args.llm_backend
    if args.llm_url:                        cfg.llm_url = args.llm_url
    if args.llm_model:                      cfg.llm_model = args.llm_model
    if args.llm_api_key:                    cfg.llm_api_key = args.llm_api_key
    if args.llm_temperature is not None:    cfg.llm_temperature = args.llm_temperature
    if args.llm_timeout is not None:        cfg.llm_timeout = args.llm_timeout
    if args.llm_debug:                      cfg.llm_debug = True

    # 开关类 (bool 取反, 默认 config.xml 是 True)
    if args.no_llm:                         cfg.use_llm = False
    if args.no_priors:                      cfg.use_priors = False
    if args.no_attribution:                 cfg.use_attribution = False
    return cfg


def run_single(args, name: str, cases, var_names, log_dir):
    from lgf.pipeline_lgf import main as lgf_main
    from lgf import llm_ops as L

    cfg = load_config_layered(args)
    # 直接走 cfg 上的非 LLM 字段 (不依赖 argparse 默认值, 让 config.xml 也能覆盖)
    cfg.population = args.population
    cfg.generations = args.generations
    cfg.seed = args.seed
    cfg.time_budget_s = args.time_budget_s
    cfg.max_edges = args.max_edges
    cfg.max_params = args.max_params
    if cfg.llm_debug:
        L.set_debug(True)
    llm = L.make_client(cfg.llm_backend, url=cfg.llm_url, model=cfg.llm_model,
                        api_key=cfg.llm_api_key,
                        temperature=cfg.llm_temperature, timeout=cfg.llm_timeout)
    if not llm.is_heuristic():
        print(f"[LLM] backend={cfg.llm_backend} url={cfg.llm_url} "
              f"model={getattr(llm, 'model', '-')} "
              f"api_key={'已设置' if getattr(llm, 'api_key', None) not in (None, '', 'EMPTY') else '未设置'}")
    t0 = time.time()
    res = lgf_main(cases=cases, var_names=var_names, cfg=cfg,
                   log_dir=log_dir, llm_client=llm, name=name)
    res["wall_s"] = time.time() - t0
    return res


def cmd_run_suite(args):
    from lgf.benchmarks import REGISTRY, make_cases, load_real_csv_simple
    suite_map = {
        "quick": ["mul_sep", "add_sep", "esat", "voce", "hollomon", "rational"],
        "all": list(REGISTRY.keys()),
    }
    names = suite_map[args.suite]
    rows = []
    for nm in names:
        bm = REGISTRY[nm]()
        cases = make_cases(bm, n_per_case=args.n_samples, noise_rel=args.noise, seed=args.seed)
        print(f"\n[{nm}] {bm.truth}")
        res = run_single(args, nm, cases, bm.var_names, args.log_dir)
        b = res["best"]
        top = res["top_final"][0]
        rows.append(dict(
            name=nm,
            truth=bm.truth,
            recovered=top["dsl"],
            loss=top["loss"],
            complexity=str(top["complexity"]),
            wall_s=res["wall_s"],
        ))
        print(f"  best loss = {top['loss']:.4e}")
        print(f"  recovered: {top['dsl']}")
        print(f"  complexity: {top['complexity']}")
        print(f"  wall_s: {res['wall_s']:.1f}")
    # 汇总表
    out_path = os.path.join(args.log_dir, f"summary_{args.suite}.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n汇总已写入", out_path)
    return rows


def cmd_run_real(args, dataset_name: str):
    from lgf.benchmarks import load_real_csv_simple
    cases, var_names = load_real_csv_simple(dataset_name)
    if not cases:
        print(f"未找到 {dataset_name} 的数据, 跳过."); return None
    print(f"\n[{dataset_name}] {var_names} (N={cases[0]['X'].shape[0]})")
    return run_single(args, f"real_{dataset_name}", cases, var_names, args.log_dir)


def main():
    args = parse_args()
    if args.list_benchmarks:
        cmd_list_benchmarks(); return
    if args.framework == "llmsr":
        print("[llmsr] 已弃用 --framework 入口. 请直接使用 "
              "D:\\AI\\LLM-Feynman\\LLM-Feynman-main\\llm_feynman\\run_experiment.py "
              "或本仓库的 LGF 框架 (--framework lgf)."); return
    os.makedirs(args.log_dir, exist_ok=True)
    if args.suite != "none":
        cmd_run_suite(args); return
    if args.real:
        cmd_run_real(args, args.real); return
    from lgf.benchmarks import REGISTRY, make_cases
    bm = REGISTRY[args.benchmark]()
    cases = make_cases(bm, n_per_case=args.n_samples, noise_rel=args.noise, seed=args.seed)
    print(f"\n[{args.benchmark}] truth: {bm.truth}")
    res = run_single(args, args.benchmark, cases, bm.var_names, args.log_dir)
    top = res["top_final"][0]
    print(f"\n  best loss = {top['loss']:.4e}")
    print(f"  recovered: {top['dsl']}")
    print(f"  complexity: {top['complexity']}")
    print(f"  wall_s: {res['wall_s']:.1f}")
    print(f"  log_dir: {args.log_dir}")


if __name__ == "__main__":
    main()