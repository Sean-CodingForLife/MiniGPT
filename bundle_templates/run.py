#!/usr/bin/env python3
"""MiniGPT Bundle v1 的独立运行器。

同一份权重可用两种引擎运行：

* ``torch`` —— PyTorch 实现（速度快，需要 torch）；
* ``pure``  —— 项目自写的 Python 张量/自动求导引擎（纯标准库，无需 PyTorch）；
* ``auto``  —— 能导入 torch 就用 torch，否则用 pure（默认）。

本文件在 bundle 目录内自包含：不需要仓库源码、不需要 transformers、不需要任何
第三方包。

用法::

    python run.py --prompt "ROMEO:"
    python run.py --engine pure --prompt "ROMEO:"
    python run.py --interactive
    python run.py --prompt "Hello" --max_new_tokens 60 --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

# NumPy 是刻意不引入的依赖，只屏蔽 PyTorch 导入时那一条相关警告。
warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

# Windows 控制台默认代码页会把中文样本打成乱码，这里统一改为 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from minigpt.tokenization.char import CharTokenizer  # noqa: E402


def _torch_available() -> bool:
    """探测 torch 是否可导入；任何导入失败都视为不可用（含缺 DLL 等环境问题）。"""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "use pure"
        return False
    return True


class TorchEngine:
    """PyTorch 引擎：延迟导入 torch，保证 pure 路径完全不需要它。"""

    name = "torch"

    def __init__(self, config, device: str, seed: int) -> None:
        """加载 ``pytorch_model.bin`` 并切到 eval 模式。

        参数:
            config: bundle 的 config.json 内容，其中 ``model_config`` 描述结构。
            device: ``auto`` / ``cpu`` / ``cuda``；auto 时有 CUDA 就用 CUDA。
            seed: 传给 ``torch.manual_seed``，固定采样随机源。
        """
        import torch

        from minigpt.models.gpt import GPT, GPTConfig

        self.torch = torch
        torch.manual_seed(seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = GPT(GPTConfig.from_dict(config["model_config"]))
        state = torch.load(ROOT / "pytorch_model.bin", map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval()

    def generate(self, prompt_ids, max_new_tokens, temperature, top_k, top_p, eos_id):
        """调用模型自身的采样接口，返回 prompt + 续写的 token id 列表。"""
        return self.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
        )


class PureEngine:
    """自写引擎：不依赖 torch，权重从 ``weights.f32`` 读取。"""

    name = "pure"

    def __init__(self, config, device: str, seed: int) -> None:
        """加载权重并创建独立随机源（pure 引擎固定跑在 CPU）。"""
        from minigpt.pure.gpt import PureGPT, PureGPTConfig
        from minigpt.pure.weights import load_weights

        weights_path = ROOT / "weights.f32"
        if not weights_path.is_file():
            raise SystemExit("bundle 缺少 weights.f32；用 export.py 重新导出，或改用 --engine torch")
        state, _index = load_weights(weights_path)
        self.model = PureGPT(PureGPTConfig.from_dict(config["model_config"]))
        self.model.load_state_dict(state)
        self.rng = random.Random(seed)
        self.device = "cpu"

    def generate(self, prompt_ids, max_new_tokens, temperature, top_k, top_p, eos_id):
        """用自写采样内核生成 token id 列表；随机性来自 ``self.rng``。"""
        return self.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            rng=self.rng,
        )


def load_bundle(engine: str, device: str, seed: int):
    """读取 config/tokenizer 并实例化指定引擎。

    返回:
        ``(config, tokenizer, implementation)``。
    异常:
        SystemExit: 指定了 torch 但环境不可用，或引擎名非法。
    """
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.load(ROOT / "tokenizer.json")
    if engine == "auto":
        engine = "torch" if _torch_available() else "pure"
    if engine == "torch":
        if not _torch_available():
            raise SystemExit("torch 不可用；请用 --engine pure（纯标准库引擎）")
        implementation = TorchEngine(config, device, seed)
    elif engine == "pure":
        implementation = PureEngine(config, device, seed)
    else:
        raise SystemExit(f"未知引擎: {engine}")
    return config, tokenizer, implementation


def answer(config, tokenizer, implementation, question: str, max_new_tokens: int, temperature, top_k, top_p) -> str:
    """按 bundle 的 prompt 模板回答一个单轮问题，返回去除特殊 token 的文本。

    采样参数的优先级：显式命令行值 > ``generation_config`` > 内置默认值。
    """
    generation = config.get("generation_config", {})
    if not max_new_tokens:
        max_new_tokens = int(generation.get("max_new_tokens", 200))
    # 命令行显式给出时优先；否则回落到 generation_config.json。
    if temperature is None:
        temperature = generation.get("temperature", 0.8)
    if top_k is None:
        top_k = generation.get("top_k", 40)
    if top_p is None:
        top_p = generation.get("top_p", 1.0)
    template = config.get("prompt_template", "{prompt}")
    prompt = template.replace("{prompt}", question)
    prompt_ids = tokenizer.encode(prompt)
    eos_id = tokenizer.state_dict().get("specials", {}).get("<eos>")
    generated = implementation.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
    )
    # 去掉 prompt 本身，只保留模型续写部分。
    return tokenizer.decode(generated[len(prompt_ids) :], skip_special=True).strip()


def build_parser() -> argparse.ArgumentParser:
    """构造 bundle 运行器参数；默认值 0/None 表示「交给 generation_config」。"""
    parser = argparse.ArgumentParser(description="Run a MiniGPT Bundle v1")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--engine", default="auto", choices=["auto", "torch", "pure"])
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行入口：单轮或交互式对话；返回进程退出码。"""
    args = build_parser().parse_args(argv)
    config, tokenizer, implementation = load_bundle(args.engine, args.device, args.seed)
    print(
        f"[bundle] {config.get('format')} stage={config.get('source', {}).get('stage')} "
        f"engine={implementation.name} device={implementation.device}"
    )
    if args.interactive:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                # Ctrl+Z / Ctrl+D：结束交互。
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            print(
                "A:",
                answer(config, tokenizer, implementation, question, args.max_new_tokens, args.temperature, args.top_k, args.top_p),
            )
        return 0
    question = args.prompt if args.prompt is not None else "Hello"
    print("A:", answer(config, tokenizer, implementation, question, args.max_new_tokens, args.temperature, args.top_k, args.top_p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
