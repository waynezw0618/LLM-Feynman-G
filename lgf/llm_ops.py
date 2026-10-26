"""lgf.llm_ops — LLM 算子层.

提供:
- LLMClient 接口与三种实现: HeuristicLLM (离线启发式), HttpFlaskLLM (本地引擎),
  OpenAICompatLLM (OpenAI 兼容 API).
- 五类外层循环指令: T_PROP / G_INIT / H_DECOMP / M_GUIDE / E_CHECK.

所有 op 都实现 graceful degradation: LLM 不可用或 JSON 解析失败时, 回退到启发式.

配置要点:
- 后端选择: `make_client(backend, url=..., model=..., api_key=...)`, backend ∈ {heuristic, flask, openai}.
- URL 语义: flask 需填到 `.../completions`; openai 只填到版本段 (`.../v1`), 自动追加 `/chat/completions`.
- api_key: 缺省按 `LGF_LLM_API_KEY` > `OPENAI_API_KEY` > `DASHSCOPE_API_KEY` 解析; 无鉴权本地服务可用默认 "EMPTY".
- 排障: 置 `LGF_LLM_DEBUG=1` (或 `set_debug(True)`) 让后端失败打印到 stderr;
  `LAST_ERROR` 保存最近一次失败详情. 不置该开关时行为与旧版一致 (静默退化为 "{}").
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .graph import Graph
from .priors import PriorConstraints, profile_summary
from .templates import TEMPLATE_REGISTRY, list_templates
from . import dsl


# ============================================================================
# 失败可见性 (反向补丁: 历史上所有后端异常都被静默吞掉, 排障困难)
# ============================================================================
# 设置环境变量 LGF_LLM_DEBUG=1 或 CLI --llm_debug 后, 失败会打印到 stderr.
DEBUG: bool = bool(os.environ.get("LGF_LLM_DEBUG", ""))

# 最近一次后端调用失败信息, 供上层 / 测试断言 (键: backend, url, error)
LAST_ERROR: Dict[str, Any] = {"backend": None, "url": None, "error": None}

# API key 环境变量查找顺序 (显式传参优先级最高)
_API_KEY_ENVS: Tuple[str, ...] = ("LGF_LLM_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY")


def _report_failure(backend: str, url: str, exc: BaseException) -> None:
    """记录并 (可选) 打印后端失败原因. 任何后端失败都不会中断主流程."""
    msg = f"{type(exc).__name__}: {exc}"
    LAST_ERROR.update(backend=backend, url=url, error=msg)
    if DEBUG:
        print(f"[LGF-LLM] backend={backend} url={url or '-'} 调用失败 -> {msg}",
              file=sys.stderr, flush=True)


def set_debug(on: bool = True) -> None:
    """运行期开关调试输出 (CLI --llm_debug 用)."""
    global DEBUG
    DEBUG = bool(on)
    if on:
        os.environ["LGF_LLM_DEBUG"] = "1"


def resolve_api_key(explicit: Optional[str] = None) -> str:
    """按 显式传参 -> 环境变量 的顺序解析 API key, 都缺失时返回 "EMPTY".

    环境变量查找顺序: LGF_LLM_API_KEY > OPENAI_API_KEY > DASHSCOPE_API_KEY.
    """
    if explicit:
        return explicit
    for name in _API_KEY_ENVS:
        val = os.environ.get(name)
        if val:
            return val
    return "EMPTY"


def _complete_one(llm: "BaseLLM", prompt: str, n: int = 1) -> str:
    """调用 LLM 并保证返回非空字符串 (任何异常/空响应都退化为 '{}').

    五个外层算子统一走这里, 避免 `llm.complete(...)[0]` 在空列表时抛 IndexError.
    """
    try:
        out = llm.complete(prompt, n=n)
    except Exception as e:                       # 客户端自身异常
        _report_failure(type(llm).__name__, getattr(llm, "url", getattr(llm, "base_url", "")), e)
        return "{}"
    if not out:
        _report_failure(type(llm).__name__, getattr(llm, "url", getattr(llm, "base_url", "")),
                        ValueError("后端返回空列表"))
        return "{}"
    return out[0] or "{}"


# ============================================================================
# LLM 客户端
# ============================================================================
class BaseLLM:
    def complete(self, prompt: str, *, n: int = 1, params: Optional[Dict] = None) -> List[str]:
        raise NotImplementedError

    def is_heuristic(self) -> bool: return False


class HeuristicLLM(BaseLLM):
    """离线启发式 LLM 模拟器, 用规则生成合理输出, 用于离线开发与基线测试."""
    def is_heuristic(self): return True

    def complete(self, prompt: str, *, n: int = 1, params: Optional[Dict] = None) -> List[str]:
        # 调用方根据 prompt 关键字路由
        kind = _infer_prompt_kind(prompt)
        if kind == "T_PROP":
            return [_templates_json()]
        if kind == "G_INIT":
            return [_initial_graphs_json(prompt)]
        if kind == "H_DECOMP":
            return [_decomp_json(prompt)]
        if kind == "M_GUIDE":
            return [_mutations_json(prompt)]
        if kind == "E_CHECK":
            return [_equivalence_json(prompt)]
        return ["{}"]


def _infer_prompt_kind(prompt: str) -> str:
    if "[T_PROP]" in prompt: return "T_PROP"
    if "[G_INIT]" in prompt: return "G_INIT"
    if "[H_DECOMP]" in prompt: return "H_DECOMP"
    if "[M_GUIDE]" in prompt: return "M_GUIDE"
    if "[E_CHECK]" in prompt: return "E_CHECK"
    return "OTHER"


class HttpFlaskLLM(BaseLLM):
    """对接外部 LLM 服务的 Flask /completions 兼容接口.

    该客户端可对接任意实现 OpenAI-ChatCompletions 之前
    `{"prompt":..., "content":[...]}` 协议的 Flask 服务 (典型如
    D:\\AI\\LLM-Feynman\\LLM-Feynman-main\\llm_feynman\\llm_engine.py).
    LGF 不依赖该文件存在, 仅在显式传入 --llm_backend flask 时才会用到.

    注意: url 会被原样 POST (仅去掉末尾 '/'), 因此**必须自带 /completions 路径**.
    api_key 可选; 若上游前置了鉴权网关, 传入后会带 Authorization: Bearer 头.
    """
    def __init__(self, url: str, api_key: Optional[str] = None,
                 temperature: float = 0.7, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def complete(self, prompt: str, *, n: int = 1, params: Optional[Dict] = None) -> List[str]:
        import requests
        body = {
            "prompt": prompt,
            "repeat_prompt": int(n),
            "params": dict(params or {}, temperature=self.temperature),
        }
        try:
            r = requests.post(self.url, json=body, timeout=self.timeout, headers=self._headers())
            if r.status_code >= 400:                       # 带上响应体, 便于定位 401/404
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            data = r.json()
            content = data.get("content")
            if not content:
                raise ValueError(f"响应缺少非空 'content' 字段, 实际 keys={list(data)[:8]}")
            return list(content)
        except Exception as e:
            _report_failure("flask", self.url, e)
            return ["{}"]


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """剥离推理模型的思考块 (如 MiniMax-M2 / DeepSeek-R1 的 <think>...</think>).

    - 完整块: 直接删除;
    - 未闭合块 (截断/只输出了思考): 从 <think> 起全部截掉, 返回空串;
    - 无思考块: 原样返回.
    这是为了防止思考文本中的花括号污染 safe_json_loads 的 `{...}` 提取.
    """
    if not text or "<think>" not in text:
        return text
    text = _THINK_BLOCK.sub("", text)
    idx = text.find("<think>")              # 残留未闭合块
    if idx >= 0:
        text = text[:idx]
    return text.strip()


class OpenAICompatLLM(BaseLLM):
    """对接 OpenAI 兼容 /chat/completions 接口 (云端 API 或本地 vLLM/Ollama/LM Studio).

    - base_url 只填到版本段 (如 https://api.openai.com/v1), 客户端自动追加 /chat/completions;
    - api_key 缺省时按 LGF_LLM_API_KEY > OPENAI_API_KEY > DASHSCOPE_API_KEY 解析,
      无鉴权的本地服务可保留默认 "EMPTY";
    - model 缺省时取 cfg.llm_model, 仍为空则回退 "gpt-3.5-turbo";
    - 推理模型输出中的 <think>...</think> 会被自动剥离 (见 _strip_reasoning).
    """
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 model: Optional[str] = None, temperature: float = 0.7,
                 timeout: float = 60.0, max_tokens: Optional[int] = None,
                 extra_body: Optional[Dict] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = resolve_api_key(api_key)
        self.model = model or "gpt-3.5-turbo"
        self.temperature = temperature
        self.timeout = timeout
        # V4 (2026-10-12): 思考模型 (GLM 5.3 / MiniMax M2.7) 的 reasoning tokens
        # 计入 max_tokens 配额 — 不显式给足时 content 会被 reasoning 吃光返回空
        # (实测 glm-5.1 简单 "hi" 花 1147 reasoning tokens). None = 不发送该字段.
        self.max_tokens = max_tokens
        # V4-b (2026-10-12): 供应商专有请求体参数 (如智谱 glm-5.3 的
        # thinking/reasoning_effort). glm-5.3 强制思考且 reasoning_effort 默认
        # max — 不显式降档时 G_INIT 实测 reasoning 8181 tokens 吃光 8192
        # 配额, content 为空 finish=length; 设 {"type":"enabled"}+low 后
        # reasoning 17 tokens, 30 候选 13s 返回, 解析率 29/30.
        self.extra_body = dict(extra_body) if extra_body else None

    @property
    def url(self) -> str:                                # 供 _report_failure 记录
        return self.base_url + "/chat/completions"

    def complete(self, prompt: str, *, n: int = 1, params: Optional[Dict] = None) -> List[str]:
        import requests
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "n": int(n),
        }
        if self.max_tokens:
            body["max_tokens"] = int(self.max_tokens)
        if self.extra_body:
            body.update(self.extra_body)
        if params:
            body.update({k: v for k, v in params.items() if k != "temperature"})
        try:
            r = requests.post(self.url, headers={"Authorization": f"Bearer {self.api_key}",
                                                 "Content-Type": "application/json"},
                              json=body, timeout=self.timeout)
            if r.status_code >= 400:                     # 401 时把 API 原始报错带出来
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            out = [_strip_reasoning(c["message"]["content"]) for c in choices
                   if isinstance(c, dict) and (c.get("message") or {}).get("content")]
            out = [s for s in out if s]     # 剥离后为空(纯思考被截断)的丢弃
            if not out:
                raise ValueError(f"'choices' 中无有效 content, 响应片段={str(data)[:200]}")
            return out
        except Exception as e:
            _report_failure("openai", self.url, e)
            return ["{}"]


def make_client(backend: str, *, url: Optional[str] = None, model: Optional[str] = None,
                api_key: Optional[str] = None,
                temperature: float = 0.7, timeout: float = 60.0,
                max_tokens: Optional[int] = None,
                extra_body: Optional[Dict] = None) -> BaseLLM:
    """按后端类型构造 LLM 客户端.

    model:
        - flask  后端忽略 (不发 model 字段);
        - openai 后端缺省回退 'gpt-3.5-turbo' (不再把 'local' 当模型名发出去).
    extra_body: 供应商专有请求体参数 (见 OpenAICompatLLM.__init__ V4-b 注释).
    """
    backend = (backend or "heuristic").lower()
    if backend == "heuristic":
        return HeuristicLLM()
    if backend == "flask":
        if not url:
            raise ValueError("flask 后端需要 --llm_url, 如 http://localhost:5000/completions")
        if "/completions" not in url:
            print(f"[LGF-LLM] 提示: flask 的 url 会被原样 POST, 通常应形如 "
                  f"http://host:port/completions (当前: {url})", file=sys.stderr, flush=True)
        return HttpFlaskLLM(url, api_key=api_key, temperature=temperature, timeout=timeout)
    if backend == "openai":
        if url and ("/chat/completions" in url or "/completions" in url):
            print(f"[LGF-LLM] 提示: openai 的 url 只需填到版本段 (如 .../v1), "
                  f"客户端会自动追加 /chat/completions (当前: {url})",
                  file=sys.stderr, flush=True)
        return OpenAICompatLLM(base_url=url or "https://api.openai.com/v1",
                               api_key=api_key, model=model,
                               temperature=temperature, timeout=timeout,
                               max_tokens=max_tokens, extra_body=extra_body)
    raise ValueError(f"未知 backend: {backend}")


# ============================================================================
# 启发式输出 (HeuristicLLM 用的 JSON 字符串)
# ============================================================================
def _templates_json() -> str:
    items = []
    for name in list_templates():
        items.append({
            "name": name,
            "kind": name,
            "desc": f"启发式提议模板 {name}",
            "dsl": "{{build_default}}",
        })
    return json.dumps({"templates": items}, ensure_ascii=False)


def _initial_graphs_json(prompt: str) -> str:
    """G_INIT: 从数据画像中提取变量名, 输出若干 DSL 候选."""
    # 从 prompt 中匹配变量名 (兼容中文 "变量:" 与英文 "variables:")
    m = re.search(r"(?:变量|variables)\s*[:：]\s*\[([^\]]+)\]", prompt)
    var_names = []
    if m:
        var_names = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    items = []
    if len(var_names) >= 2:
        items.append({"dsl": f"{var_names[0]} + {var_names[1]}", "name": "g_lin"})
        items.append({"dsl": f"{var_names[0]} * {var_names[1]}", "name": "g_mul"})
        items.append({"dsl": f"({var_names[0]})^2 + ({var_names[1]})^2", "name": "g_sq"})
    elif var_names:
        v = var_names[0]
        items.append({"dsl": f"{v}", "name": "g_id"})
        items.append({"dsl": f"{v}^2", "name": "g_sq"})
        items.append({"dsl": f"exp({v})", "name": "g_exp"})
        items.append({"dsl": f"sin({v})", "name": "g_sin"})
        items.append({"dsl": f"({v}) / (1 + ({v}))", "name": "g_sat"})
    return json.dumps({"graphs": items}, ensure_ascii=False)


def _decomp_json(prompt: str) -> str:
    return json.dumps({"decompositions": []}, ensure_ascii=False)


def _mutations_json(prompt: str) -> str:
    # 启发式: 输出一个随机扰动参数 + 一个换模板的建议
    return json.dumps({"mutations": [
        {"type": "perturb_param", "param_id": "C1", "magnitude": 0.2},
        {"type": "replace_subtree", "path": [], "new_dsl": "exp(1/(C2*x1 + C3))"},
    ]}, ensure_ascii=False)


def _equivalence_json(prompt: str) -> str:
    return json.dumps({"classes": [], "ranking": []}, ensure_ascii=False)


# ============================================================================
# 通用 JSON 解析 (健壮)
# ============================================================================
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def safe_json_loads(text: str, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not text:
        return default
    # V4 (2026-10-12): 剥离 ```json ... ``` 代码围栏 (GLM/MiniMax 常见包裹,
    # 旧版靠 _JSON_BLOCK 正则兜底, 但围栏内的尾随逗号/注释仍会导致整体解析失败)
    if "```" in text:
        text = re.sub(r"```(?:json|JSON|javascript|js)?", "", text)
    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass
    # 从第一对花括号截取
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return default


# ============================================================================
# 五个外层循环 op
# ============================================================================
def T_PROP(llm: BaseLLM, profile: Dict, prior_constraints: PriorConstraints,
           existing_templates: Sequence[str]) -> List[Dict]:
    """提议新模板 (返回 dict 列表, 由调用方决定如何注册)."""
    prompt = f"""[T_PROP] 你是方程发现助手. 基于数据画像与已有模板, 提议最多 3 个新结构模板.
变量: {profile['var_names']}
数据画像:
{profile_summary(profile)}
已有模板: {list(existing_templates)}
请仅输出 JSON: {{\"templates\": [{{"name": str, "kind": str, "desc": str, "dsl": str}}]}}.
dsl 字段支持 $1, $2 占位槽位; P#(x)=... 多项式; C#/K# 参数. 无新模板时输出空数组."""
    text = _complete_one(llm, prompt)
    obj = safe_json_loads(text, {"templates": []})
    out = []
    for t in obj.get("templates", []):
        if not t.get("dsl"):
            continue
        out.append(dict(name=t.get("name") or "t" + str(len(out)+1),
                       dsl=t["dsl"], desc=t.get("desc", "")))
    if not out:
        return []
    return out


def G_INIT_wall_hint(profile: Dict) -> str:
    """P1d-B2: 壁面特征约束行 — 让 LLM 初始候选一半以上含 yw.

    P1c 教训: T1 minimax 全程不含 yw (冷启动就没引入, 事后 PIN 拦不住),
    近壁阻尼通道缺失. 这里在 G_INIT prompt 里显式要求.

    V4-b (2026-10-12): 追加对数律 (law of the wall) 提示 —
    槽道流 Re_τ=180 无明显对数层, 但 b23 的近壁行为仍可借 log(yw) 类
    结构表达 (yw->0 发散, 与饱和/旋转因子组合后给出正确剖面形状).
    前几轮实验证实 sqrt(roloc)·(C·log(yw)+C'·roloc) 型可分离乘积 + 对数律
    因子是最强家族 (LGF minimax 0.085, PySR 0.033); 只提示家族方向,
    不喂具体公式, 留给 LLM 推导.
    """
    _vn = [str(v) for v in (profile.get("var_names") or [])]
    if not any(v in _vn for v in ("yw", "ypd")):
        return ""
    return ("约束: yw 是无量纲真壁面距离 (yw=0 壁面, 1 中心), 近壁阻尼 φ(yw) 物理必需. "
            "你生成的候选中至少一半必须显式包含 yw 项, 形式上可考虑:\n"
            "  (a) 阻尼型: exp(-yw/τ) 或 1/(C#+yw) 或 1/(1+(yw/τ)^2) (饱和/衰减);\n"
            "  (b) 对数律型: log(yw) 或 C·log(yw) (与饱和/旋转因子组合抑制 yw->0 发散);\n"
            "  (c) 嵌套比值: roloc·log(roloc)/yw 或 sqrt(roloc)·(C·log(yw)+C'·roloc) "
            "型可分离乘积 (前几轮实验证实的强家族).")


def G_INIT(llm: BaseLLM, profile: Dict, templates: Sequence[str], n: int = 8,
           *, use_fewshot: bool = True, extra_context: str = "") -> List[str]:
    """生成 n 个初始图 DSL (字符串列表).

    v3a-A3: use_fewshot=True 时注入 few-shot 合法 DSL 示例 (默认开).
    true-ASM (2026-10-09): extra_context 非空时注入任务物理上下文
    (张量基/不变量/Re_t/Ro_loc 含义与结构约束), 让 LLM 带物理先验出候选.
    """
    fewshot = ("\n[FEW-SHOT 合法 DSL 范例, 直接复用其算子/结构]\n"
               '  {"dsl": "(C4 * (1)) / (K1 * (1) + C3 * (eta2)) + C2 * (zeta2e)", "name": "rational_plus_linear"}\n'
               '  {"dsl": "C1 * (tanh(inv(C3 * (s2o2e) + C2 * (1))))", "name": "saturation_tanh"}\n'
               '  {"dsl": "C1 * (exp(inv(C2 * (yw) + K1 * (1))))", "name": "wall_damp"}\n'
               '  {"dsl": "C1 * (sqrt(roloc)) * (C2 * (log(yw)) + C3 * (roloc))", "name": "log_law_separable"}\n'
               '  {"dsl": "C1 * (roloc) * (log(roloc)) / (yw + C2 * (1))", "name": "nested_ratio"}\n'
               "注意: 变量名只能用上述已知特征名; 算子名只能用白名单; 系数写 C1/C2/K1 等."
               ) if use_fewshot else ""
    ctx_block = (f"\n[TASK PHYSICS CONTEXT (任务物理背景, 候选必须尊重)]\n{extra_context}\n"
                 ) if extra_context else ""
    prompt = f"""[G_INIT] 你是方程发现助手. 基于数据画像, 生成 {n} 个候选方程.
变量: {profile['var_names']}
数据画像:
{profile_summary(profile)}
可用模板: {list(templates)}
{G_INIT_wall_hint(profile)}{ctx_block}{fewshot}
请输出 JSON: {{"graphs": [{{"dsl": str, "name": str}}, ...]}}.
dsl 须符合: add mul sub div exp log sin cos tanh sqrt inv neg, 系数 C#, K#, 多项式 P#(x)=..., 幂 x^2..4.
    硬规则: 只允许上述算子名, 自造函数名 (如 sat/esat/gauss/voce/holl/mul2 等) 会解析失败被整条丢弃;
    需要饱和/衰减行为时用 tanh(C#*x), C#/(K#+x), exp(C#/x) 显式组合表达;
    需要对数律近壁行为时用 log(C#*yw + K#) 或 C#*log(yw) 组合饱和/旋转因子表达 (单 log(yw) 在 yw->0 发散, 须配饱和)."""
    text = _complete_one(llm, prompt)
    obj = safe_json_loads(text, {"graphs": []})
    out = []
    for g in obj.get("graphs", []):
        d = g.get("dsl")
        if isinstance(d, str) and d.strip():
            out.append(d)
    return out[:n]


def H_DECOMP(llm: BaseLLM, profile: Dict, prior_constraints: PriorConstraints,
             top_graphs: Sequence[Tuple[Graph, float]]) -> List[Dict]:
    prompt = f"""[H_DECOMP] 你是方程发现助手. 基于先验与 top-K 图, 建议 1-2 个分解假设.
先验: {prior_constraints.to_dict()}
top-K loss: {[f'{g[0].name}={g[1]:.4f}' for g in top_graphs[:3]]}
输出 JSON: {{\"decompositions\": [{{\"type\": str, \"vars\": [int,...], \"hint\": str}}]}}."""
    text = _complete_one(llm, prompt)
    obj = safe_json_loads(text, {"decompositions": []})
    return list(obj.get("decompositions", []))


def M_GUIDE(llm: BaseLLM, top_graphs: Sequence[Tuple[Graph, float]],
           attribution_report: Dict, extra_context: str = "") -> List[Dict]:
    """基于归因生成定向变异指令. 每条: {'type': str, ...}.

    L1 物理注解 + L2 壁面区误差分布注入到 prompt, 让 LLM 看到
    "yw (壁面距离) 不是普通变量, 删除会丢失近壁精度" 的物理意义.
    true-ASM (2026-10-09): extra_context 注入任务物理上下文,
    引导变异尊重张量基/不变量结构 (K# 逐工况常数会被系统拒绝).
    """
    top_dsls = [(g.to_dsl() if hasattr(g, "to_dsl") else str(g)) for g, _ in top_graphs[:3]]
    feature_roles = attribution_report.get("feature_roles") or {}
    ypd_bin_errors = attribution_report.get("ypd_bin_errors") or []

    # L1: 物理角色块
    phys_lines = []
    if feature_roles:
        phys_lines.append("PHYSICAL CONTEXT (特征物理角色, 不要把 🔒PIN 特征全删):")
        for name, info in feature_roles.items():
            pin = "🔒PIN" if str(info.get("pinned", "false")).lower() == "true" else "  "
            phys_lines.append(f"  {pin} {name}: role={info.get('role','?')}, "
                              f"desc={info.get('desc','')}")

    # L2: 壁面区误差分布
    wall_lines = []
    if ypd_bin_errors:
        wall_lines.append("WALL-REGION ERROR DISTRIBUTION (wall-distance bins, "
                          "yw=0 at wall):")
        for e in ypd_bin_errors:
            wall_lines.append(f"  bin {e['bin']}: error_share={e['share']:.1%}, "
                              f"mean_|resid|={e['mean_abs_resid']:.3g}, n={e['n']}")
        near = sum(e["share"] for e in ypd_bin_errors if e["bin"].startswith("[0,"))
        if near > 0.4:
            wall_lines.append(f"  ⚠ 近壁 (yw<15) 误差占 {near:.1%} → 不要删除所有 yw 节点")

    # L3 隐式: LLM 返回的 replace_subtree 若把 🔒PIN 特征全删, 会被 post-filter 拦截
    pin_names = [n for n, info in feature_roles.items()
                 if str(info.get("pinned", "false")).lower() == "true"]
    pin_warning = ""
    if pin_names:
        pin_warning = (f"FORBIDDEN: replace_subtree 操作把以下 🔒PIN 特征全部删除 → "
                      f"会被自动回滚到原图, 请保留至少 1 个 {pin_names} 节点: {pin_names}")

    # P1d-A2/A3/B3: 硬规则块 (目标值域 + 危险分母 + 最差工况 + 目标级禁用算子)
    target_sem = attribution_report.get("target_semantics") or {}
    per_case = attribution_report.get("per_case_summary") or []
    hard_lines = []
    if target_sem:
        y_min, y_max = target_sem.get("y_min"), target_sem.get("y_max")
        frac_neg = target_sem.get("frac_neg")
        if y_min is not None and y_max is not None and frac_neg is not None:
            if frac_neg < 1e-3:
                hard_lines.append(
                    f"- 目标 y ∈ [{y_min:.3g}, {y_max:.3g}] 且恒非负: new_dsl 在全部 "
                    f"工况的取值都必须保持该值域, sub(减法)/div(分数) 结构高风险, 除非 "
                    f"能证明不会产生负值或爆炸")
            elif frac_neg > 0.99:
                hard_lines.append(
                    f"- 目标 y ∈ [{y_min:.3g}, {y_max:.3g}] 且恒非正: 注意保持符号")
        for name, frac in (target_sem.get("vanishing") or {}).items():
            hard_lines.append(
                f"- 特征 {name} 在 {frac:.0%} 的训练点上 |x|≈0: 禁止把 {name} 放在分母 "
                f"(div/inv), 也不可作为减法主导项 — 会在这些点上爆炸")
    if per_case:
        try:
            worst = max(per_case, key=lambda p: p["nmse"])
            if worst["nmse"] > 0.3:
                hard_lines.append(
                    f"- 最差工况 {worst['name']} NMSE={worst['nmse']:.3g} "
                    f"({worst['label']}): 变异必须优先修复该工况, 不得以恶化它为代价")
        except Exception:
            pass
    forbidden = list(attribution_report.get("forbidden_ops") or [])
    if forbidden:
        hard_lines.append(
            f"- STRUCTURAL PRIOR: 本目标禁用算子 {forbidden} — 含这些算子的 new_dsl "
            f"会被直接丢弃; 请用 mul / powi(x^2..4) / exp / 加法 等单调形式")
    hard_block = ""
    if hard_lines:
        hard_block = ("HARD RULES (违反即被系统丢弃或回滚):\n"
                      + "\n".join(hard_lines))

    ctx_block = ""
    if extra_context:
        ctx_block = (f"TASK PHYSICS CONTEXT (任务物理背景, 变异必须尊重):\n"
                     f"{extra_context}\n")

    prompt = f"""[M_GUIDE] 你是方程发现助手. 基于归因报告给出定向变异.
top 图 DSL: {top_dsls[:3]}
归因报告: {attribution_report.get('text_summary', '')}
{chr(10).join(phys_lines)}
{chr(10).join(wall_lines)}
{pin_warning}
{ctx_block}{hard_block}
建议变异:
- replace_subtree(path: list[int], new_dsl: str)
- perturb_param(param_id: str, magnitude: float)
输出 JSON: {{\"mutations\": [{{\"type\": str, ...}}, ...]}}."""
    text = _complete_one(llm, prompt)
    obj = safe_json_loads(text, {"mutations": []})
    return list(obj.get("mutations", []))


def E_CHECK(llm: BaseLLM, top_graphs: Sequence[Tuple[Graph, float]],
           cases) -> List[Dict]:
    prompt = f"""[E_CHECK] 你是方程发现助手. top-K 图中可能数学等价, 给出结构选择.
top DSL: {[g.to_dsl() if hasattr(g, 'to_dsl') else str(g) for g, _ in top_graphs[:5]]}
输出 JSON: {{\"ranking\": [\"name1\", \"name2\", ...]}}."""
    text = _complete_one(llm, prompt)
    obj = safe_json_loads(text, {"ranking": []})
    return list(obj.get("ranking", []))


# ============================================================================
# 将 op 输出落地到图 (DSL → Graph, 异常丢弃)
# ============================================================================
def parse_dsl_safely(text: str, var_names: Sequence[str]) -> Optional[Graph]:
    try:
        return dsl.parse_dsl(text, var_names)
    except Exception:
        return None


def _has_op(root, ops) -> bool:
    """图节点级算子扫描 (P1d-B3): 子树中是否含任一禁用算子."""
    from .graph import walk
    return any(n.op in ops for _, n in walk(root))


def apply_mutations(graph: Graph, mutations: Sequence[Dict], rng,
                   forbidden_ops: Sequence[str] = ()) -> Graph:
    """轻量落地: 仅支持 perturb_param 和简单的 replace_subtree.

    任何修改后若新图结构非法 (例如 add 子节点数 < 2), 立即回退到原图,
    防止污染种群. 这样 LLM 即使返回畸形 DSL 也不会让整个 pipeline 崩.

    L3 物理约束: 若 replace_subtree 把某个 🔒PIN 特征(如 yw 壁面距离) 全部删除,
    且原图含该特征, 则**回滚到原图**, 防止 LLM 把壁面信息通道打掉.

    P1d-B3: forbidden_ops (目标级禁用算子, 如 T5g 的 sub/div) —
    新子树含任一禁用算子即静默丢弃该变异, 把结构先验从 prompt 软约束升级为硬约束.

    V3-d (2026-10-08): 全部变异都未实际生效时返回 None (旧版返回原图 clone,
    导致 pipeline 的 `ng is not None` 检查形同虚设 — 原图以 loss=inf 重新混入
    种群, 污染而非引导). 生效判定: perturb_param 命中过至少一个参数, 或
    replace_subtree 成功挂载过一次.
    """
    g = graph.clone()
    applied_any = False
    # 记录原图含哪些 PIN 特征 (用 var_names 子串扫描 + var op 计数)
    PIN_NAMES = ("yw", "ypd")   # 壁面距离特征 (yw 新, ypd 旧兼容), 未来可扩展
    def _count_pin(g_: Graph) -> int:
        c = 0
        from .graph import walk
        for _, n in walk(g_.root):
            if n.op == "var" and 0 <= n.attrs.get("index", -1) < len(g_.var_names):
                if g_.var_names[n.attrs["index"]] in PIN_NAMES:
                    c += 1
        return c
    original_pin_count = _count_pin(g)
    FORBIDDEN = set(forbidden_ops or ())
    for m in mutations or []:
        if not isinstance(m, dict):
            continue
        t = m.get("type")
        if t == "perturb_param":
            pid = m.get("param_id")
            # V3-e (2026-10-09): LLM 返回的 magnitude 为 NaN/inf/<=0/非数值时,
            # rng.normal(0, mag) 抛 ValueError → M_GUIDE 后期停摆 (真 ASM 跑批
            # v/g/e 三个 mg 臂分别在 gen19/34/39 触发). 一律回退默认 0.2.
            try:
                mag = float(m.get("magnitude", 0.2))
            except (TypeError, ValueError):
                mag = 0.2
            if not (np.isfinite(mag) and mag > 0.0):
                mag = 0.2
            for p in g.params():
                if p.name == pid:
                    # 爆参 (NaN/inf init) 时以 0 为基重置, 防止 NaN 静默入种群
                    base = p.init if np.isfinite(p.init) else 0.0
                    p.init = float(base + rng.normal(0, mag))
                    if abs(p.init) < 1e-6: p.init = 1e-3
                    applied_any = True
        elif t == "replace_subtree":
            new_dsl = m.get("new_dsl")
            if not new_dsl: continue
            new_g = parse_dsl_safely(new_dsl, g.var_names)
            if new_g is None: continue
            # P1d-B3: 目标级禁用算子 (图节点级扫描, 不做文本匹配避免误杀)
            if FORBIDDEN and _has_op(new_g.root, FORBIDDEN):
                continue
            # L3 PIN 保护: 若原图含 PIN 特征, 新图把 PIN 删光, 直接放弃此次变异
            if original_pin_count > 0 and _count_pin(new_g) == 0:
                continue   # 静默丢弃, 让 GP 重抽
            # 试改并校验; 失败回滚到原图, 不污染种群.
            backup_root = g.root
            try:
                g.root = new_g.root
                g._simplify()
                g._normalize_params()
                g._validate()                       # ← 新增校验
                applied_any = True
            except Exception:
                g.root = backup_root                # ← 回滚
    if not applied_any:
        return None                                 # V3-d: 全失败 → None, 不污染种群
    return g