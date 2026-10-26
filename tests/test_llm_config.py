"""LLM 后端配置面单元测试.

覆盖补丁点:
1. api_key 解析顺序 (显式 > LGF_LLM_API_KEY > OPENAI_API_KEY > DASHSCOPE_API_KEY > "EMPTY")
2. openai 后端 model 缺省回退 (不再把 'local' 当模型名发出)
3. url 语义 (openai 自动追加 /chat/completions; flask 原样 POST)
4. 失败可见性 (HTTP 4xx 带响应体; LAST_ERROR; --llm_debug 开关; _complete_one 兜底非空)
5. flask 端到端 (自带 mock 服务, 验证 T_PROP 可解析)
6. LGFConfig 新字段

运行: python -W ignore tests/test_llm_config.py
"""
import sys, os, json, time, threading, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler, HTTPServer

from lgf import llm_ops
from lgf.llm_ops import (make_client, resolve_api_key, set_debug, _complete_one,
                         HeuristicLLM, HttpFlaskLLM, OpenAICompatLLM)
from lgf.config import LGFConfig

PASS = 0; FAIL = 0; FAILED = []


def check(name, fn):
    global PASS, FAIL
    t0 = time.time()
    try:
        fn()
        PASS += 1
        print(f"[PASS] {name}  ({time.time()-t0:.1f}s)")
    except Exception:
        FAIL += 1; FAILED.append(name)
        print(f"[FAIL] {name}")
        traceback.print_exc()


# ============================================================================
# 环境变量隔离工具 (避免污染本机真实 key)
# ============================================================================
_KEY_ENVS = ("LGF_LLM_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY")


class _CleanEnv:
    """进入时清除全部 API key 环境变量, 退出时恢复."""
    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in _KEY_ENVS}
        for k in _KEY_ENVS:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *a):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


# ============================================================================
# 1. api_key 解析
# ============================================================================
def test_api_key_explicit_wins():
    with _CleanEnv():
        os.environ["LGF_LLM_API_KEY"] = "from-env"
        assert resolve_api_key("from-arg") == "from-arg"


def test_api_key_env_priority():
    with _CleanEnv():
        os.environ["OPENAI_API_KEY"] = "openai-key"
        os.environ["DASHSCOPE_API_KEY"] = "dash-key"
        assert resolve_api_key() == "openai-key", "OPENAI 应优先于 DASHSCOPE"
        os.environ["LGF_LLM_API_KEY"] = "lgf-key"
        assert resolve_api_key() == "lgf-key", "LGF_LLM_API_KEY 优先级最高"


def test_api_key_fallback_empty():
    with _CleanEnv():
        assert resolve_api_key() == "EMPTY"


def test_client_picks_up_env_key():
    with _CleanEnv():
        os.environ["LGF_LLM_API_KEY"] = "sk-test-123"
        c = make_client("openai", url="https://api.openai.com/v1", model="gpt-4o-mini")
        assert c.api_key == "sk-test-123", c.api_key


# ============================================================================
# 2. model 缺省回退 (回归: 旧版默认 'local' 会被当模型名发出)
# ============================================================================
def test_openai_model_fallback():
    c = OpenAICompatLLM(base_url="https://x/v1", model=None)
    assert c.model == "gpt-3.5-turbo", c.model
    c2 = make_client("openai", url="https://x/v1", model=None)
    assert c2.model == "gpt-3.5-turbo", c2.model
    c3 = make_client("openai", url="https://x/v1", model="qwen-plus")
    assert c3.model == "qwen-plus"


def test_flask_ignores_model():
    """flask 协议不发 model 字段, 因此传什么 model 都不影响."""
    c = make_client("flask", url="http://localhost:5050/completions", model=None)
    assert isinstance(c, HttpFlaskLLM)
    assert not hasattr(c, "model")


# ============================================================================
# 3. url 语义
# ============================================================================
def test_openai_url_composition():
    c = OpenAICompatLLM(base_url="https://api.openai.com/v1/")
    assert c.url == "https://api.openai.com/v1/chat/completions", c.url


def test_flask_url_preserved():
    c = HttpFlaskLLM("http://localhost:5050/completions/")
    assert c.url == "http://localhost:5050/completions", c.url


def test_flask_requires_url():
    try:
        make_client("flask")
    except ValueError as e:
        assert "llm_url" in str(e)
        return
    raise AssertionError("flask 后端缺 url 时应抛 ValueError")


def test_unknown_backend_raises():
    try:
        make_client("nope")
    except ValueError as e:
        assert "nope" in str(e)
        return
    raise AssertionError("未知 backend 应抛 ValueError")


# ============================================================================
# 4. 失败可见性
# ============================================================================
def test_failure_is_recorded():
    """不可达端口: 客户端不抛异常, 但 LAST_ERROR 必须记录."""
    llm_ops.LAST_ERROR.update(backend=None, url=None, error=None)
    c = HttpFlaskLLM("http://127.0.0.1:1/completions", timeout=1.0)
    out = c.complete("hi")
    assert out == ["{}"], out
    assert llm_ops.LAST_ERROR["backend"] == "flask", llm_ops.LAST_ERROR
    assert llm_ops.LAST_ERROR["error"], "应记录异常文本"


def test_complete_one_never_empty():
    """_complete_one 对任何客户端都返回非空字符串 (旧版 [0] 会 IndexError)."""
    class _Empty:
        is_heuristic = lambda self: False          # noqa: E731
        def complete(self, prompt, *, n=1, params=None):
            return []
    assert _complete_one(_Empty(), "x") == "{}"


def test_complete_one_recovers_real_output():
    class _Two:
        def complete(self, prompt, *, n=1, params=None):
            return ["first", "second"]
    assert _complete_one(_Two(), "x") == "first"


def test_complete_one_swallows_exception():
    class _Boom:
        url = "http://boom/completions"
        def complete(self, prompt, *, n=1, params=None):
            raise RuntimeError("kaboom")
    assert _complete_one(_Boom(), "x") == "{}"
    assert "kaboom" in (llm_ops.LAST_ERROR["error"] or "")


def test_http_status_reported_with_body():
    """HTTP 401 时错误信息应包含状态码与响应体 (旧版只 raise_for_status, 无 body)."""
    httpd = HTTPServer(("127.0.0.1", 0), _AuthErrorHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/completions"
        out = HttpFlaskLLM(url, timeout=3.0).complete("hi")
        assert out == ["{}"], out
        err = llm_ops.LAST_ERROR["error"] or ""
        assert "401" in err, err
        assert "Invalid API key" in err, err
    finally:
        httpd.shutdown(); httpd.server_close()


def test_debug_toggle():
    old = llm_ops.DEBUG
    try:
        set_debug(True)
        assert llm_ops.DEBUG is True
        assert os.environ.get("LGF_LLM_DEBUG") == "1"
        set_debug(False)
        assert llm_ops.DEBUG is False
    finally:
        llm_ops.DEBUG = old


def test_heuristic_never_touches_network():
    c = make_client("heuristic")
    assert isinstance(c, HeuristicLLM)
    assert c.is_heuristic()
    assert c.complete("[T_PROP] whatever")           # 非空


# ============================================================================
# 5. flask 端到端 (自带 mock)
# ============================================================================
def test_flask_end_to_end_t_prop():
    httpd = HTTPServer(("127.0.0.1", 0), _MockLLMHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/completions"
        c = make_client("flask", url=url, timeout=5.0)
        text = _complete_one(c, "[T_PROP] 请提议模板")
        obj = llm_ops.safe_json_loads(text, {"templates": []})
        assert obj.get("templates"), f"T_PROP 应返回模板列表, 实际: {text[:200]}"
    finally:
        httpd.shutdown(); httpd.server_close()


def test_flask_end_to_end_g_init():
    httpd = HTTPServer(("127.0.0.1", 0), _MockLLMHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/completions"
        c = make_client("flask", url=url, timeout=5.0)
        text = _complete_one(c, "[G_INIT] 变量: ['x1', 'x2']")
        obj = llm_ops.safe_json_loads(text, {"graphs": []})
        assert obj.get("graphs"), f"G_INIT 应返回图列表, 实际: {text[:200]}"
        assert len(obj["graphs"]) >= 3
    finally:
        httpd.shutdown(); httpd.server_close()


# ============================================================================
# 6. LGFConfig 新字段
# ============================================================================
def test_config_has_new_fields():
    cfg = LGFConfig()
    assert getattr(cfg, "llm_api_key", "MISSING") is None
    assert cfg.llm_debug is False
    assert cfg.llm_model is None, "默认应为 None 以触发后端各自回退"
    d = cfg.to_dict()
    for k in ("llm_api_key", "llm_debug", "llm_model", "llm_url",
              "llm_temperature", "llm_timeout", "llm_backend"):
        assert k in d, f"to_dict 缺字段 {k}"


def test_pipeline_fallback_client_uses_config():
    """pipeline_lgf 在 llm_client=None 时应按 cfg.llm_api_key / llm_debug 构造客户端."""
    from lgf.pipeline_lgf import main as lgf_main
    import inspect
    src = inspect.getsource(lgf_main)
    assert "cfg.llm_api_key" in src, "pipeline 未把 llm_api_key 透传给 make_client"
    assert "set_debug" in src, "pipeline 未处理 llm_debug"


# ============================================================================
# mock 服务
# ============================================================================
class _MockLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(ln).decode("utf-8", "replace") or "{}")
        except Exception:
            req = {}
        prompt = req.get("prompt", "")
        kind = llm_ops._infer_prompt_kind(prompt)
        dispatch = {
            "T_PROP": lambda p: llm_ops._templates_json(),
            "G_INIT": llm_ops._initial_graphs_json,
            "H_DECOMP": llm_ops._decomp_json,
            "M_GUIDE": llm_ops._mutations_json,
            "E_CHECK": llm_ops._equivalence_json,
        }
        fn = dispatch.get(kind)
        content = fn(prompt) if fn else "{}"
        payload = json.dumps({"content": [content]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _AuthErrorHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(ln)
        payload = b'{"error":{"message":"Invalid API key provided"}}'
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("LLM 后端配置面测试")
    print("=" * 72)

    check("api_key: 显式传参优先",             test_api_key_explicit_wins)
    check("api_key: 环境变量优先级",           test_api_key_env_priority)
    check("api_key: 全部缺失回退 EMPTY",       test_api_key_fallback_empty)
    check("api_key: 客户端读取环境变量",        test_client_picks_up_env_key)
    check("model: openai 缺省回退",            test_openai_model_fallback)
    check("model: flask 忽略 model",           test_flask_ignores_model)
    check("url: openai 自动追加路径",           test_openai_url_composition)
    check("url: flask 原样保留路径",            test_flask_url_preserved)
    check("url: flask 缺 url 报错",            test_flask_requires_url)
    check("backend: 未知后端报错",              test_unknown_backend_raises)
    check("失败可见性: LAST_ERROR 记录",        test_failure_is_recorded)
    check("失败可见性: _complete_one 非空",     test_complete_one_never_empty)
    check("失败可见性: _complete_one 取首条",    test_complete_one_recovers_real_output)
    check("失败可见性: _complete_one 吞异常",    test_complete_one_swallows_exception)
    check("失败可见性: 4xx 带状态码与响应体",     test_http_status_reported_with_body)
    check("调试开关: set_debug 切换",           test_debug_toggle)
    check("heuristic 不发网络请求",             test_heuristic_never_touches_network)
    check("端到端: flask mock T_PROP",         test_flask_end_to_end_t_prop)
    check("端到端: flask mock G_INIT",         test_flask_end_to_end_g_init)
    check("配置: LGFConfig 新字段",            test_config_has_new_fields)
    check("配置: pipeline 透传 llm_api_key",    test_pipeline_fallback_client_uses_config)

    print("=" * 72)
    print(f"PASS={PASS}  FAIL={FAIL}  TOTAL={PASS+FAIL}")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
    print("ALL TESTS PASSED")
