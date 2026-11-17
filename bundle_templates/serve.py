#!/usr/bin/env python3
"""MiniGPT Bundle v1 的极简 HTTP 服务（HTTP 层仅用标准库）。

用法::

    python serve.py --port 8000
    curl -X POST http://127.0.0.1:8000/api/generate -H "Content-Type: application/json" \
         -d '{"prompt":"Hello","max_new_tokens":40}'

不依赖任何 Web 框架；但模型推理仍需要 PyTorch（若要用纯标准库引擎，请用
``run.py --engine pure``）。本服务**没有鉴权**，默认只绑定 127.0.0.1。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

import torch

# 统一控制台编码，避免 Windows 下中文输出乱码。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from minigpt.models.gpt import GPT, GPTConfig  # noqa: E402
from minigpt.tokenization.char import CharTokenizer  # noqa: E402

# 进程启动时一次性加载配置/词表/权重，请求处理阶段只做前向计算。
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
TOKENIZER = CharTokenizer.load(ROOT / "tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = GPT(GPTConfig.from_dict(CONFIG["model_config"]))
STATE = torch.load(ROOT / "pytorch_model.bin", map_location=DEVICE, weights_only=True)
MODEL.load_state_dict(STATE)
MODEL.to(DEVICE).eval()
EOS_ID = TOKENIZER.state_dict().get("specials", {}).get("<eos>")
TEMPLATE = CONFIG.get("prompt_template", "{prompt}")
GEN = CONFIG.get("generation_config", {})

# 资源上限：本服务没有鉴权，必须限制单个请求，避免一个客户端耗尽 CPU/内存
# （超大请求体、无上限生成长度等）。
MAX_BODY_BYTES = 1_000_000
MAX_PROMPT_CHARS = 8_000
MAX_NEW_TOKENS_CAP = 512
SOCKET_TIMEOUT_S = 120.0


def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> str:
    """套用 bundle 的 prompt 模板生成文本，返回去除特殊 token 的续写部分。"""
    text = TEMPLATE.replace("{prompt}", prompt)
    ids = TOKENIZER.encode(text)
    generated = MODEL.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=EOS_ID,
    )
    return TOKENIZER.decode(generated[len(ids) :], skip_special=True).strip()


class Handler(BaseHTTPRequestHandler):
    """单路由处理器：``GET /health``、``GET /``、``POST /api/generate``。"""

    def log_message(self, format, *args):
        """覆盖默认日志格式，统一加 ``[http]`` 前缀并输出到标准输出。"""
        print("[http] " + format % args)

    def setup(self):
        """为每个连接设置 socket 超时，防止慢客户端长期占用线程。"""
        super().setup()
        self.connection.settimeout(SOCKET_TIMEOUT_S)

    def _json(self, status: int, payload: dict) -> None:
        """发送 JSON 响应（UTF-8、显式 Content-Length）。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """健康检查与根路径说明；其它路径统一 404。"""
        if self.path == "/health":
            self._json(200, {"status": "ok", "device": DEVICE, "stage": CONFIG.get("source", {}).get("stage")})
        elif self.path == "/":
            self._json(200, {"service": "MiniGPT bundle", "endpoints": ["/health", "POST /api/generate"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        """生成接口：校验请求体大小、prompt 长度与采样参数后同步返回结果。"""
        if self.path != "/api/generate":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                self._json(400, {"error": "Content-Length 非法"})
                return
            if length > MAX_BODY_BYTES:
                self._json(413, {"error": f"请求体过大（>{MAX_BODY_BYTES} 字节）"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                self._json(400, {"error": "请求体必须是 JSON 对象"})
                return
            prompt = str(payload.get("prompt", ""))
            if len(prompt) > MAX_PROMPT_CHARS:
                self._json(400, {"error": f"prompt 过长（>{MAX_PROMPT_CHARS} 字符）"})
                return
            max_new_tokens = int(payload.get("max_new_tokens", GEN.get("max_new_tokens", 100)))
            if not 0 <= max_new_tokens <= MAX_NEW_TOKENS_CAP:
                self._json(400, {"error": f"max_new_tokens 必须在 [0, {MAX_NEW_TOKENS_CAP}]"})
                return
            temperature = float(payload.get("temperature", GEN.get("temperature", 0.8)))
            top_k = int(payload.get("top_k", GEN.get("top_k", 40)))
            top_p = float(payload.get("top_p", GEN.get("top_p", 1.0)))
            if temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
                self._json(400, {"error": "采样参数越界（temperature>=0, top_k>=0, 0<top_p<=1）"})
                return
            self._json(200, {"prompt": prompt, "text": generate(prompt, max_new_tokens, temperature, top_k, top_p)})
        except Exception as error:
            self._json(400, {"error": f"{type(error).__name__}: {error}"})


def main() -> int:
    """启动 HTTP 服务；Ctrl+C 退出并关闭监听套接字。"""
    parser = argparse.ArgumentParser(description="Serve a MiniGPT Bundle v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MiniGPT bundle: http://{args.host}:{server.server_address[1]}/  (device={DEVICE})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
