"""用 SFT checkpoint 进行多轮对话。

模型训练时使用的模板是 ``Q: {question}\\nA: {answer}<eos>``。多轮对话把历史
Q/A 轮次按同一模板拼接进上下文；超出 ``block_size`` 时从最旧轮次开始丢弃，
始终保留最新问题。交互命令：``/reset``、``/help``、``/system <text>``。
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  （把项目根目录加入 sys.path）
from minigpt.config import overlay_config
from minigpt.data.formatters import format_sft_prompt, normalize_answer, normalize_question
from minigpt.utils import ensure_hash_seed, seed_everything


class Conversation:
    """多轮上下文管理：历史拼接 + 超长截断（丢最旧轮次，保最新问题）。"""

    def __init__(self, tokenizer, block_size: int, system: str = ""):
        """初始化对话状态。

        参数:
            tokenizer: 与模型配套的 tokenizer，用于按 token 数判断是否超长。
            block_size: 模型最大上下文长度（token 数）。
            system: 可选的系统前缀，会加在整段对话最前面。
        """
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.system = system.strip()
        self.turns: list[tuple[str, str]] = []

    def add_turn(self, question: str, answer: str) -> None:
        """追加一轮已完成的问答；空白问答会被忽略。"""
        q = normalize_question(question)
        a = normalize_answer(answer)
        if q and a:
            self.turns.append((q, a))

    def reset(self) -> None:
        """清空全部历史轮次（保留系统前缀）。"""
        self.turns.clear()

    def set_system(self, text: str) -> None:
        """设置系统前缀；传入空串等价于清除。"""
        self.system = text.strip()

    def _render(self, turns: list[tuple[str, str]], question: str) -> str:
        """把系统前缀、历史轮次与当前问题渲染成一段纯文本 prompt。"""
        parts: list[str] = []
        if self.system:
            parts.append(self.system)
        for q, a in turns:
            parts.append(f"Q: {q}\nA: {a}")
        parts.append(format_sft_prompt(question))
        return "\n".join(parts)

    def build_prompt(self, question: str) -> str:
        """构造不超过 ``block_size`` 的完整 prompt。

        优先丢弃最旧的历史轮次；若只剩当前问题仍然超长，则按 token 截断并
        只保留尾部，保证模型至少能看到最新问题与 ``A:`` 前缀。
        """
        question = normalize_question(question)
        if not question:
            raise ValueError("question 不能为空")
        turns = list(self.turns)
        while True:
            prompt = self._render(turns, question)
            ids = self.tokenizer.encode(prompt)
            if len(ids) <= self.block_size:
                return prompt
            if turns:
                turns.pop(0)
                continue
            # 丢完历史仍超长（单问过长）：保留尾部 token（最新问题与 A: 前缀）。
            # 不能把 <unk> 等特殊 token 还原成字面量再编码——字面量占多个字符，
            # 截断窗口切在字面量中间时会来回振荡不收敛。这里只保留普通字符
            # token（一词元一字符），解码后长度必然 <= block_size，无需迭代。
            tail = ids[-self.block_size :]
            specials = getattr(self.tokenizer, "_special_ids", frozenset())
            kept = [token_id for token_id in tail if token_id not in specials]
            return self.tokenizer.decode(kept, skip_special=True)


def parse_command(text: str) -> tuple[str, str] | None:
    """识别交互命令；普通输入返回 None。

    返回:
        ``(命令名, 参数)``，命令名属于 ``reset`` / ``help`` / ``system``；
        不认识的 ``/xxx`` 也返回 None，按普通问题处理。
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    name, _, arg = stripped[1:].partition(" ")
    if name in {"reset", "help", "system"}:
        return name, arg.strip()
    return None


def _load_bundle(checkpoint: str, device: str):
    """统一加载分派（minigpt.inference.load_bundle 按文件头识别 pure/torch）。"""

    from minigpt.inference import load_bundle

    return load_bundle(checkpoint, device=device)


def _answer_prompt(bundle, prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> str:
    """对一段完整 prompt 生成答案，只返回新生成的文本（不含 prompt 本身）。"""
    prompt_ids = bundle.tokenizer.encode(prompt)
    eos_id = getattr(bundle.tokenizer, "eos_id", None)
    generated = bundle.backend.generate(
        bundle.model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
    )
    # generate 返回 prompt + 续写，这里按 prompt 长度切出续写部分。
    continuation = generated[len(prompt_ids) :]
    return bundle.tokenizer.decode(continuation, skip_special=True).strip()


def _answer(bundle, question: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> str:
    """单轮问答：套用 SFT 模板后生成答案。"""
    return _answer_prompt(bundle, format_sft_prompt(question), max_new_tokens, temperature, top_k, top_p)


def _topk_text(bundle, topk: list[dict]) -> str:
    """把 top-k 候选渲染成 ``token 概率 | token 概率`` 的一行文本。"""
    return " | ".join(
        f"{bundle.tokenizer.token_repr(item['token_id'])} {item['prob']:.3f}"
        for item in topk
    )


def _stream_answer(bundle, question: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float, trace_top_k: int) -> str:
    """单轮流式问答：把问题套进 SFT 模板后转交流式生成。"""
    return _stream_answer_prompt(
        bundle, format_sft_prompt(question), max_new_tokens, temperature, top_k, top_p, trace_top_k
    )


def _stream_answer_prompt(bundle, prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float, trace_top_k: int) -> str:
    """逐 token 打印生成过程（采样概率 + top-k 候选），返回完整答案文本。

    该函数只负责展示与拼接；采样逻辑在后端 ``generate_stream`` 中，保证与
    非流式路径使用同一套随机数与温度/top-k/top-p 语义。
    """
    prompt_ids = bundle.tokenizer.encode(prompt)
    eos_id = getattr(bundle.tokenizer, "eos_id", None)
    continuation: list[int] = []
    steps = 0
    for item in bundle.backend.generate_stream(
        bundle.model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
        trace_top_k=trace_top_k,
    ):
        steps += 1
        token = bundle.tokenizer.token_repr(item["token_id"])
        print(
            f"  {item['step']:>3}  {token:<12} p={item['prob']:.4f}  {_topk_text(bundle, item['topk'])}",
            flush=True,
        )
        if item["finished"]:
            break
        continuation.append(item["token_id"])
    if steps == 0:
        print("  (no tokens: max_new_tokens=0)", flush=True)
    return bundle.tokenizer.decode(continuation, skip_special=True).strip()


def build_parser() -> argparse.ArgumentParser:
    """构造对话命令行参数（公共配置项由 overlay_config 补齐）。"""
    parser = argparse.ArgumentParser(description="Chat with a MiniGPT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None, help="single-turn question; omit with --interactive")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--system", default="", help="system prefix prepended to the conversation context")
    parser.add_argument("--stream", action="store_true", help="print every generated token with its probability and top-k candidates")
    parser.add_argument("--trace_top_k", type=int, default=5, help="how many candidates to show per streamed token")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    return parser


HELP_TEXT = (
    "命令：/reset 清空对话历史 | /system <text> 设置系统前缀 | /help 显示帮助 | exit 退出"
)


def main(argv: list[str] | None = None) -> int:
    """对话入口：加载 checkpoint，进入单轮或交互式多轮对话。

    返回:
        进程退出码：0 正常结束，1 当前后端不支持流式输出。
    """
    args = overlay_config(
        build_parser(),
        argv,
        stage="sft",
        field_map={"prompt": "sample_prompt"},
    )
    ensure_hash_seed(args.seed)
    seed_everything(args.seed)
    bundle = _load_bundle(args.checkpoint, args.device)
    print(f"[chat] stage={bundle.payload.get('stage')} device={bundle.backend.device_info().get('device')}")
    if args.stream and not hasattr(bundle.backend, "generate_stream"):
        print("[chat] this backend does not support --stream", file=sys.stderr)
        return 1

    block_size = int(getattr(bundle.model.config, "block_size", 256))
    conv = Conversation(bundle.tokenizer, block_size=block_size, system=args.system)

    def answer(question: str) -> str:
        """按当前模式（普通/流式）回答一个问题，使用会话拼好的 prompt。"""
        prompt = conv.build_prompt(question)
        if args.stream:
            print(f"[stream] Q: {question}", flush=True)
            return _stream_answer_prompt(
                bundle,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                args.trace_top_k,
            )
        return _answer_prompt(bundle, prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p)

    if args.interactive:
        print("多轮对话已就绪。" + HELP_TEXT)
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                # Ctrl+Z / Ctrl+D 结束输入，视作正常退出。
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            command = parse_command(question)
            if command is not None:
                name, arg = command
                if name == "reset":
                    conv.reset()
                    print("[chat] 历史已清空")
                elif name == "help":
                    print(HELP_TEXT)
                elif name == "system":
                    conv.set_system(arg)
                    print(f"[chat] 系统前缀已设置：{arg!r}" if arg else "[chat] 系统前缀已清除")
                continue
            reply = answer(question)
            conv.add_turn(question, reply)
            print("A:", reply)
        return 0
    # 非交互模式：--prompt 缺省时用 "Hello"，方便快速冒烟测试。
    question = args.prompt if args.prompt is not None else "Hello"
    print("A:", answer(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
