"""纯 Python 版 GPT：与 ``minigpt.models.gpt`` 结构完全一致，但不依赖 PyTorch。

结构（与训练模型逐项对应）::

    token embedding + 可学习绝对位置编码
      -> L x [Pre-LN -> 因果多头自注意力（QKV 融合）-> 残差
              Pre-LN -> MLP（4 倍宽，GELU-tanh）-> 残差]
      -> 最终 LayerNorm
      -> LM head（与 token embedding 绑定）

参数命名刻意对齐 PyTorch 的 ``state_dict``（``transformer.wte.weight``、
``transformer.h.0.attn.c_attn.weight`` 等），因此 checkpoint 可按名字直接转换，
无需猜测映射表。Dropout 在 torch 模型中仅在训练时生效，这里直接关闭
（推理/评估语义）。

本模块只导入标准库。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

from minigpt.pure import tensor as T

NEG_INF = float("-inf")


@dataclass
class PureGPTConfig:
    """纯 Python 前向所需的模型配置子集。"""

    block_size: int
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PureGPTConfig":
        """从字典构造，忽略未知字段（兼容 checkpoint 里的扩展配置）。"""
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def from_flat_config(cls, config: Any, vocab_size: int) -> "PureGPTConfig":
        """从项目 ``Config`` 对象构造（字段与 torch 后端使用的完全一致）。"""

        return cls(
            block_size=int(config.block_size),
            vocab_size=int(vocab_size),
            n_layer=int(config.n_layer),
            n_head=int(config.n_head),
            n_embd=int(config.n_embd),
            dropout=float(getattr(config, "dropout", 0.0)),
            bias=bool(getattr(config, "bias", False)),
            tie_weights=bool(getattr(config, "tie_weights", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        """导出为可 JSON 序列化的字典（写入 checkpoint 的 model_config）。"""
        return {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
            "tie_weights": self.tie_weights,
        }


def _normal(rng: random.Random, count: int, std: float) -> list[float]:
    """用给定随机源生成 ``count`` 个 N(0, std) 样本（确定性初始化用）。"""
    return [rng.gauss(0.0, std) for _ in range(count)]


class PureGPT:
    """手写前向的迷你 GPT，参数为可参与自动求导的 :class:`Tensor`。"""

    def __init__(self, config: PureGPTConfig, seed: int | None = None) -> None:
        """校验结构约束并初始化全部参数。

        参数:
            config: 模型结构。
            seed: 初始化随机种子；相同 seed 得到完全相同的初始权重。
        异常:
            ValueError: ``n_embd`` 不能被 ``n_head`` 整除，或请求了 pure 引擎
            不支持的 ``bias=True`` / ``tie_weights=False``。
        """
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if config.bias:
            raise ValueError("the pure engine does not support bias=True; use the torch backend")
        if not config.tie_weights:
            raise ValueError("the pure engine requires tie_weights=True; use the torch backend")
        self.config = config
        rng = random.Random(seed)
        width = config.n_embd
        self.params: dict[str, T.Tensor] = {}

        def add(name: str, values: list[float], shape: Sequence[int]) -> None:
            self.params[name] = T.Tensor(values, shape, requires_grad=True)

        add("transformer.wte.weight", _normal(rng, config.vocab_size * width, 0.02), (config.vocab_size, width))
        add("transformer.wpe.weight", _normal(rng, config.block_size * width, 0.02), (config.block_size, width))
        for layer in range(config.n_layer):
            prefix = f"transformer.h.{layer}"
            add(f"{prefix}.ln_1.weight", [1.0] * width, (width,))
            add(f"{prefix}.attn.c_attn.weight", _normal(rng, 3 * width * width, 0.02), (3 * width, width))
            residual_std = 0.02 / math.sqrt(2 * max(config.n_layer, 1))
            add(f"{prefix}.attn.c_proj.weight", _normal(rng, width * width, residual_std), (width, width))
            add(f"{prefix}.ln_2.weight", [1.0] * width, (width,))
            add(f"{prefix}.mlp.c_fc.weight", _normal(rng, 4 * width * width, 0.02), (4 * width, width))
            add(f"{prefix}.mlp.c_proj.weight", _normal(rng, 4 * width * width, residual_std), (width, 4 * width))
        add("transformer.ln_f.weight", [1.0] * width, (width,))

    # -- parameters -----------------------------------------------------
    def named_parameters(self):
        """按 ``(名称, 张量)`` 迭代参数（名称即 state_dict 键）。"""
        return self.params.items()

    def parameters(self) -> list[T.Tensor]:
        """返回参数列表（优化器按此顺序持有动量）。"""
        return list(self.params.values())

    def num_parameters(self) -> int:
        """总参数量（绑定权重只计一次）。"""
        return sum(tensor.numel for tensor in self.params.values())

    def zero_grad(self) -> None:
        """清空所有参数梯度。"""
        for tensor in self.params.values():
            tensor.zero_grad()

    def state_dict(self) -> dict[str, tuple[tuple[int, ...], list[float]]]:
        """导出 ``名称 -> (形状, 扁平数据)``（深拷贝，可安全序列化）。"""
        return {name: (tensor.shape, list(tensor.data)) for name, tensor in self.params.items()}

    def load_state_dict(
        self,
        state: dict[str, tuple[Sequence[int], Sequence[float]]],
        strict: bool = True,
    ) -> list[str]:
        """加载 ``名称 -> (形状, 数据)``；返回被跳过的键（如 bias/tied head）。

        ``strict=True`` 时，未知参数与缺失参数都会抛错——截断的 checkpoint
        不允许静默保留随机权重。
        """

        skipped: list[str] = []
        for name, (shape, values) in state.items():
            if name.startswith("lm_head.") or name.endswith(".bias"):
                # 绑定输出头 / 无 bias 模型：torch checkpoint 可能带有这些键。
                skipped.append(name)
                continue
            if name not in self.params:
                if strict:
                    raise KeyError(f"unknown parameter in state_dict: {name}")
                skipped.append(name)
                continue
            expected = self.params[name].shape
            if tuple(int(size) for size in shape) != expected:
                raise ValueError(f"shape mismatch for {name}: {tuple(shape)} != {expected}")
            if len(values) != self.params[name].numel:
                raise ValueError(f"value count mismatch for {name}")
            self.params[name].data = [float(value) for value in values]
        missing = [name for name in self.params if name not in state]
        if missing and strict:
            raise KeyError(f"missing parameters: {missing}")
        return skipped

    # -- forward --------------------------------------------------------
    def _mask(self, length: int) -> T.Tensor:
        """返回 ``(length, length)`` 的因果掩码（上三角为 -inf），并按长度缓存。"""
        # 因果掩码只依赖长度，跨前向/逐 token 生成复用同一份（无梯度、不被修改），
        # 避免每次前向都重建 T×T 列表。
        cache = getattr(self, "_mask_cache", None)
        if cache is None:
            cache = {}
            self._mask_cache = cache
        cached = cache.get(length)
        if cached is None:
            values: list[float] = []
            for row in range(length):
                for column in range(length):
                    values.append(0.0 if column <= row else NEG_INF)
            cached = T.Tensor(values, (length, length))
            cache[length] = cached
        return cached

    def forward(
        self,
        idx: Sequence[Sequence[int]],
        targets: Sequence[Sequence[int]] | None = None,
        ignore_index: int = -100,
        record_attention: bool = False,
    ) -> tuple[T.Tensor, T.Tensor | None]:
        """返回 ``(logits (B,T,V), loss 标量或 None)``。

        参数:
            idx: ``(B, T)`` token id。
            targets: 与 ``idx`` 同形状的目标；给出时计算交叉熵。
            ignore_index: 目标中的忽略值（默认 -100）。
            record_attention: 为 True 时把每层注意力与最终隐层暂存到
                ``self.last_attention`` / ``self.last_hidden``（生成追踪用）；
                默认关闭，训练/评估热路径不付出额外拷贝。
        """

        batch = len(idx)
        if batch == 0:
            raise ValueError("empty batch")
        length = len(idx[0])
        if any(len(row) != length for row in idx):
            raise ValueError("ragged batch")
        if length > self.config.block_size:
            raise ValueError(f"sequence length {length} exceeds block_size {self.config.block_size}")
        width = self.config.n_embd
        head_dim = width // self.config.n_head

        token_ids = [int(token) for row in idx for token in row]
        x = T.embedding(self.params["transformer.wte.weight"], token_ids)
        position_ids = list(range(length)) * batch
        x = T.add(x, T.embedding(self.params["transformer.wpe.weight"], position_ids))

        mask = self._mask(length)
        if record_attention:
            self.last_attention = []
            self.last_hidden = None
        for layer in range(self.config.n_layer):
            prefix = f"transformer.h.{layer}"
            normed = T.layer_norm(x, self.params[f"{prefix}.ln_1.weight"])
            qkv = T.matmul(normed, T.transpose_last(self.params[f"{prefix}.attn.c_attn.weight"]))
            q = T.slice_last(qkv, 0, width)
            k = T.slice_last(qkv, width, 2 * width)
            v = T.slice_last(qkv, 2 * width, 3 * width)
            q = T.permute(T.reshape(q, (batch, length, self.config.n_head, head_dim)), (0, 2, 1, 3))
            k = T.permute(T.reshape(k, (batch, length, self.config.n_head, head_dim)), (0, 2, 1, 3))
            v = T.permute(T.reshape(v, (batch, length, self.config.n_head, head_dim)), (0, 2, 1, 3))

            scores = T.scale(T.matmul(q, T.transpose_last(k)), 1.0 / math.sqrt(head_dim))
            scores = T.add(scores, mask)
            weights = T.softmax(scores)
            if record_attention:
                self.last_attention.append(_nested_view(weights, (batch, self.config.n_head, length, length)))
            attended = T.matmul(weights, v)
            attended = T.reshape(T.permute(attended, (0, 2, 1, 3)), (batch * length, width))
            projected = T.matmul(attended, T.transpose_last(self.params[f"{prefix}.attn.c_proj.weight"]))
            x = T.add(x, projected)

            normed = T.layer_norm(x, self.params[f"{prefix}.ln_2.weight"])
            hidden = T.matmul(normed, T.transpose_last(self.params[f"{prefix}.mlp.c_fc.weight"]))
            hidden = T.gelu(hidden)
            hidden = T.matmul(hidden, T.transpose_last(self.params[f"{prefix}.mlp.c_proj.weight"]))
            x = T.add(x, hidden)

        x = T.layer_norm(x, self.params["transformer.ln_f.weight"])
        if record_attention:
            self.last_hidden = _nested_view(x, (batch, length, width))
        # 绑定权重的 LM head：logits = x @ wte^T
        logits = T.matmul(x, T.transpose_last(self.params["transformer.wte.weight"]))
        logits = T.reshape(logits, (batch, length, self.config.vocab_size))
        loss = None
        if targets is not None:
            flat_targets = [int(value) for row in targets for value in row]
            loss = T.cross_entropy(
                T.reshape(logits, (batch * length, self.config.vocab_size)),
                flat_targets,
                ignore_index=ignore_index,
            )
        return logits, loss

    # -- generation -----------------------------------------------------
    def _last_logits(self, prompt_ids: Sequence[int]) -> list[float]:
        """在 ``no_grad`` 下前向，只取最后一个位置的 logits。"""
        with T.no_grad():
            logits, _ = self.forward([list(prompt_ids)])
        width = self.config.vocab_size
        start = (len(prompt_ids) - 1) * width
        return logits.data[start : start + width]

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        eos_id: int | None = None,
        rng: random.Random | None = None,
    ) -> list[int]:
        """贪心 / top-k / top-p 采样，规则与 torch 后端保持一致。"""

        if rng is None:
            rng = random
        if not prompt_ids:
            raise ValueError("prompt must not be empty")
        output = [int(token) for token in prompt_ids]
        for _ in range(max_new_tokens):
            logits = self._last_logits(output[-self.config.block_size :])
            token_id, _probs = _next_token(logits, temperature, top_k, top_p, rng)
            if eos_id is not None and token_id == eos_id:
                break  # 与 torch 一致：遇到 EOS 终止，且不把 EOS 追加进结果
            output.append(token_id)
        return output

    def generate_stream(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        eos_id: int | None = None,
        trace_top_k: int = 0,
        rng: random.Random | None = None,
        trace_attention: bool = False,
        trace_sampling: bool = False,
        trace_candidates: int = 20,
    ):
        """逐 token 产出字典（step/prob/topk/finished）。

        ``trace_attention=True`` 附带 ``attention``（[layer][head][T][T]）与
        ``hidden``（末位置隐层向量）；``trace_sampling=True`` 附带 ``sampling``
        （与 torch 版同一份字段契约）。两者都关闭时走原路径，随机流不变。
        """

        if not prompt_ids:
            raise ValueError("prompt must not be empty")
        if rng is None:
            rng = random
        from minigpt import sampling as _sampling

        context = [int(token) for token in prompt_ids]
        for step in range(1, int(max_new_tokens) + 1):
            window = context[-self.config.block_size :]
            attention = None
            hidden = None
            if trace_attention:
                with T.no_grad():
                    logits_tensor, _ = self.forward([window], record_attention=True)
                width = self.config.vocab_size
                start = (len(window) - 1) * width
                logits = logits_tensor.data[start : start + width]
                attention = self.last_attention
                hidden = self.last_hidden[-1]
            else:
                logits = self._last_logits(window)
            sampling_trace: dict[str, Any] | None = None
            if trace_sampling:
                draw = None if temperature <= 0 else rng.random()
                token_id, probs, sampling_trace = _sampling.sample_with_trace(
                    logits, temperature, top_k, top_p, draw,
                    trace_candidates=trace_candidates,
                )
            else:
                token_id, probs = _next_token(logits, temperature, top_k, top_p, rng)
            topk: list[dict[str, Any]] = []
            if trace_top_k and trace_top_k > 0:
                order = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)[:trace_top_k]
                topk = [
                    {"token_id": index, "prob": probs[index]}
                    for index in order
                ]
            finished = eos_id is not None and token_id == eos_id
            item: dict[str, Any] = {
                "step": step,
                "token_id": token_id,
                "prob": probs[token_id],
                "finished": finished,
                "context_len": len(context),
                "topk": topk,
            }
            if trace_attention:
                item["attention"] = attention
                item["hidden"] = hidden
            if sampling_trace is not None:
                item["sampling"] = sampling_trace
            yield item
            if finished:
                break
            context.append(token_id)


class _FlatRows:
    """扁平数据的行视图：``obj[i]`` 返回第 i 行（浮点列表），支持切片。"""

    __slots__ = ("_data", "_rows", "_cols")

    def __init__(self, data: list[float], rows: int, cols: int) -> None:
        """包装扁平数据；``rows``/``cols`` 描述行数与每行宽度。"""
        self._data = data
        self._rows = int(rows)
        self._cols = int(cols)

    def __len__(self) -> int:
        """行数。"""
        return self._rows

    def __getitem__(self, index):
        """按行取浮点列表；支持切片与负索引。"""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._rows))]
        if index < 0:
            index += self._rows
        start = index * self._cols
        return self._data[start : start + self._cols]


def _nested_view(tensor: T.Tensor, shape: Sequence[int]):
    """把张量转成追踪契约形状（去掉 batch 维，仅支持 batch=1）。

    * (1, head, T, T) -> ``[head][T][T]``（头列表，元素为行视图）；
    * (1, T, width)   -> ``[T][width]``（行视图）。
    与 torch 追踪的 ``attn[0].tolist()`` / ``hidden[0, -1].tolist()`` 对齐。
    """

    dims = tuple(int(size) for size in shape)
    data = list(tensor.data)
    if len(dims) == 4:
        batch, heads, rows, cols = dims
        if batch != 1:
            raise ValueError("trace only supports batch=1")
        stride = rows * cols
        return [_FlatRows(data[index * stride : (index + 1) * stride], rows, cols) for index in range(heads)]
    if len(dims) == 3:
        batch, rows, cols = dims
        if batch != 1:
            raise ValueError("trace only supports batch=1")
        return _FlatRows(data, rows, cols)
    raise ValueError(f"unsupported trace shape {dims}")


def _next_token(
    logits: Sequence[float],
    temperature: float,
    top_k: int,
    top_p: float,
    rng: random.Random,
) -> tuple[int, list[float]]:
    """采样一个 token，返回 ``(token_id, 最终概率列表)``。

    规则统一委托 minigpt.sampling 内核（与 torch 后端同一份实现）。贪心路径
    不消费随机数；采样路径先取 draw 再选词，随机数流与旧实现一致。
    """
    from minigpt import sampling

    if temperature <= 0:
        probs = sampling.softmax(logits)
        token_id = max(range(len(probs)), key=lambda index: probs[index])
        return token_id, probs
    draw = rng.random()
    return sampling.sample_from_logits(logits, temperature, top_k, top_p, draw)


__all__ = ["PureGPT", "PureGPTConfig"]
