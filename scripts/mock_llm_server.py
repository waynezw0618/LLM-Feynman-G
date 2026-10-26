"""Mock LLM 服务器 (用于无 GPU/无 LLM 时验证 HTTP 客户端路径).

模拟 LLM-Feynman 兼容的 /completions 接口 (原始参考:
D:\\AI\\LLM-Feynman\\LLM-Feynman-main\\llm_feynman\\llm_engine.py),
用启发式响应. 该 mock 是 LGF 自带的开发工具, 不依赖 llm_feynman 包.

启动:  python scripts/mock_llm_server.py --port 5050
调用:  python main.py --framework lgf --benchmark esat --llm_backend flask \
                       --llm_url http://localhost:5050/completions
"""
from __future__ import annotations
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# 让 mock server 可以 import lgf.llm_ops 的启发式函数
sys.path.insert(0, ".")

from lgf.llm_ops import (
    _templates_json, _initial_graphs_json, _decomp_json,
    _mutations_json, _equivalence_json, _infer_prompt_kind,
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(ln).decode("utf-8", errors="replace")
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        prompt = req.get("prompt", "")
        n = int(req.get("repeat_prompt", 1))
        kind = _infer_prompt_kind(prompt)
        if kind == "T_PROP":
            content = _templates_json()
        elif kind == "G_INIT":
            content = _initial_graphs_json(prompt)
        elif kind == "H_DECOMP":
            content = _decomp_json(prompt)
        elif kind == "M_GUIDE":
            content = _mutations_json(prompt)
        elif kind == "E_CHECK":
            content = _equivalence_json(prompt)
        else:
            content = "{}"
        out = {"content": [content] * n}
        data = json.dumps(out, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args = p.parse_args()
    addr = (args.host, args.port)
    httpd = HTTPServer(addr, Handler)
    print(f"Mock LLM server listening on http://{args.host}:{args.port}/completions")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()