"""用 PyTorch 基础算子从零实现的小型 GPT。

有意**不**使用 ``torch.nn.Transformer``：项目要讲清楚的每个机制都直接可见：

    token ids (B, T)
      -> token embedding + learned absolute position embedding
      -> L x [Pre-LN -> causal multi-head self-attention -> residual
              Pre-LN -> MLP (4x, GELU) -> residual]
      -> final LayerNorm
      -> LM head (tied with the token embedding)
      -> logits (B, T, vocab_size)

形状约定
--------
* ``idx``: ``(B, T)`` int64 token ids，``T <= block_size``
* ``targets``: ``(B, T)`` int64，值为 ``-100`` 的位置不参与 loss
* ``logits``: ``(B, T, vocab_size)``
* 注意力中间量: ``(B, n_head, T, T_total)``，``T_total`` 含 KV cache 的历史

与 :mod:`minigpt.pure.gpt` 的纯标准库实现保持结构同构，便于逐层对照数值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """GPT 的架构超参（与 ``tokenizer`` 无关的纯结构描述）。

    字段:
        block_size: 最大上下文长度（位置嵌入表大小），前向时 ``T`` 不得超过它。
        vocab_size: 词表大小，决定嵌入表与 LM head 的输出维度。
        n_layer: Transformer block 层数。
        n_head: 注意力头数；``n_embd`` 必须能被它整除。
        n_embd: 残差流宽度（hidden size）。
        dropout: 嵌入/注意力/残差/MLP 各项 dropout 的统一概率。
        bias: Linear 与 LayerNorm 是否使用偏置项。
        tie_weights: 是否让 LM head 与 token embedding 共享权重。
    """

    block_size: int
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True

    @classmethod
    def from_flat_config(cls, config: Any, vocab_size: int) -> "GPTConfig":
        """从扁平的 ``Config`` 对象提取模型超参。

        参数:
            config: 任意带 ``block_size/n_layer/n_head/n_embd/dropout/bias/
                tie_weights`` 属性的对象（通常是 ``minigpt.config.Config``）。
            vocab_size: 词表大小，由 tokenizer 决定，故不从 ``config`` 读取。
        返回:
            新建的 :class:`GPTConfig`。
        """
        return cls(
            block_size=config.block_size,
            vocab_size=vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            tie_weights=config.tie_weights,
        )

    def to_dict(self) -> dict[str, Any]:
        """导出为纯 Python 字典，用于写入 checkpoint 的 ``model_config`` 段。"""
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPTConfig":
        """从字典恢复配置；只取认识的键，忽略多出来的字段。

        参数:
            data: 通常来自 checkpoint 的 ``model_config``，可能含旧版本遗留键。
        返回:
            新建的 :class:`GPTConfig`，缺失键使用数据类默认值。
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class CausalSelfAttention(nn.Module):
    """多头自注意力，使用严格下三角掩码（位置 t 只能看到 <= t 的键）。

    支持两种调用方式：整段前向（训练/无缓存推理）与带 ``past_kv`` 的增量
    前向（逐 token 解码），两者共用同一套权重与掩码。
    """

    def __init__(self, config: GPTConfig):
        """按配置创建 qkv 融合投影、输出投影与 dropout 层。

        参数:
            config: 模型超参；``n_embd`` 必须能被 ``n_head`` 整除。
        异常:
            ValueError: ``n_embd % n_head != 0``（否则 ``head_dim`` 无整数解）。
        """
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # 单次融合投影是 GPT 的常见实现细节：kernel 调用更少、参数统计更直观。
        # 数学上等价于分开的 q/k/v 三个 Linear。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # persistent=False：掩码是推导出来的、不可学习，模型创建时重建即可，
        # 因此无需写进 checkpoint（也避免不同 block_size 之间加载冲突）。
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        """计算注意力输出。

        参数:
            x: ``(B, T, C)`` 输入隐藏状态。
            return_attn: 为 True 时额外返回 softmax 后的注意力权重（可视化用）。
            past_kv: 上一步的 ``(key, value)`` 缓存，形状 ``(B, H, T_past, D)``；
                ``None`` 表示从零开始。
            use_cache: 为 True 时返回新的 ``(key, value)`` 供下一步使用。
        返回:
            ``(y, attn_or_None, present_or_None)``，其中 ``y`` 为
            ``(B, T, C)``，``attn`` 为 ``(B, H, T, T_total)``，
            ``present`` 为拼接后的 ``(B, H, T_total, D)``。
        异常:
            ValueError: 新序列长度或缓存总长度超过 ``block_size``。
        """
        B, T, C = x.shape
        block_size = self.causal_mask.shape[-1]
        if T > block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {block_size}")

        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        total = k.size(2)
        if total > block_size:
            raise ValueError(f"cached sequence length {total} exceeds block_size {block_size}")

        # 缩放点积注意力：除以 sqrt(head_dim) 把点积方差拉回 O(1)，防止 softmax 饱和。
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, total)
        # 因果掩码：新 token 位于绝对位置 past_len..past_len+T-1，
        # 只能看到不晚于自身位置的键。
        mask = self.causal_mask[:, :, past_len : past_len + T, :total]
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        weights = att
        att = self.attn_dropout(att)

        y = att @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        present = (k, v) if use_cache else None
        return y, (weights if return_attn else None), present


class MLP(nn.Module):
    """Transformer 前馈网络：Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout。"""

    def __init__(self, config: GPTConfig):
        """创建两层投影；隐层宽度固定为 ``4 * n_embd``。"""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, C)`` -> ``(B, T, C)`` 的逐位置非线性变换。"""
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm 的 Transformer block。

    残差连接外的两个子层各自先做 LayerNorm（Pre-LN），相比 Post-LN 更易训练、
    不需要学习率 warmup 也能稳定。
    """

    def __init__(self, config: GPTConfig):
        """创建 ``ln_1 -> attn -> residual -> ln_2 -> mlp -> residual`` 结构。"""
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        """依次执行注意力子层与 MLP 子层，并原样透传注意力/缓存结果。

        参数/返回: 语义同 :meth:`CausalSelfAttention.forward`，额外包含 ``x``
        的残差更新；返回 ``(x, attn_or_None, present_or_None)``。
        """
        attn_out, attn, present = self.attn(
            self.ln_1(x), return_attn=return_attn, past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn, present


class GPT(nn.Module):
    """decoder-only GPT：可学习绝对位置、Pre-LN、可选权重共享。"""

    def __init__(self, config: GPTConfig):
        """构建嵌入、若干 Block、最终 LayerNorm 与 LM head，并初始化权重。

        参数:
            config: 架构超参；``tie_weights=True`` 时 LM head 与 ``wte`` 共享
                同一张量（state_dict 中只出现一次）。
        """
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            # 共享同一 Parameter 对象：反向传播时两份梯度自动累加。
            self.lm_head.weight = self.transformer["wte"].weight

        self.apply(self._init_weights)
        # GPT-2 式残差缩放：让残差流方差随深度大致保持恒定。
        # 只对输出投影（注意力 c_proj 与 MLP c_proj）生效，按 sqrt(2*n_layer) 缩小。
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """``nn.Module.apply`` 的初始化回调：Linear/Embedding 用 N(0, 0.02)，偏置置零。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """返回参数量（共享权重只计一次，用于文档中的模型规模统计）。"""
        # parameters() 会对共享（tied）参数去重，因此 LM head 只计一次；
        # 这个数字与手册里的参数统计口径一致。
        return sum(p.numel() for p in self.parameters())

    def _forward_impl(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
        return_attns: bool = False,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        """所有前向路径的公共实现（训练、可视化、增量解码都走这里）。

        参数:
            idx: ``(B, T)`` token id。
            targets: ``(B, T)`` 标签；``None`` 时不算 loss。
            ignore_index: 不参与 loss 的标签值（默认 -100，用于屏蔽 prompt）。
            return_attns: 是否收集每层注意力矩阵。
            past_kv: 每层一个 ``(key, value)`` 缓存，长度须等于 ``n_layer``。
            use_cache: 是否返回本步的 KV 缓存。
        返回:
            ``(logits, loss_or_None, hidden, attns, presents_or_None)``，其中
            ``logits`` 为 ``(B, T, vocab_size)``、``hidden`` 为最终 LN 后的
            ``(B, T, C)``（embedding 追踪用），``attns`` 长度 ``n_layer``。
        异常:
            ValueError: ``idx`` 不是二维，或 ``T`` 超过 ``block_size``。
        """
        if idx.ndim != 2:
            raise ValueError(f"idx must be (B, T), got {tuple(idx.shape)}")
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        # 位置偏移：增量解码时新 token 的绝对位置从缓存长度开始。
        past_len = past_kv[0][0].size(2) if past_kv else 0
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        x = self.transformer["drop"](self.transformer["wte"](idx) + self.transformer["wpe"](pos))

        attns: list[torch.Tensor] = []
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, block in enumerate(self.transformer["h"]):
            layer_past = past_kv[layer] if past_kv else None
            x, attn, present = block(
                x, return_attn=return_attns, past_kv=layer_past, use_cache=use_cache
            )
            if return_attns:
                attns.append(attn)
            if use_cache:
                presents.append(present)
        x = self.transformer["ln_f"](x)
        hidden = x
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 展平成 (B*T, vocab) 与 (B*T,)，ignore_index 位置由 cross_entropy 跳过。
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss, hidden, attns, (presents if use_cache else None)

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        """增量前向：返回 (logits, loss, present_kv)。

        present_kv 是每层的 (key, value) 缓存，可直接作为下一次调用的 past_kv。
        位置编码按缓存长度自动偏移；总长度不得超过 block_size。
        """
        logits, loss, _, _, present = self._forward_impl(idx, past_kv=past_kv, use_cache=True)
        return logits, loss, present

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
        return_attns: bool = False,
    ):
        """标准前向：返回 ``(logits, loss)`` 或 ``(logits, loss, attns)``。

        参数:
            idx: ``(B, T)`` token id。
            targets: ``(B, T)`` 标签；``None`` 时 loss 为 ``None``。
            ignore_index: 不参与 loss 的标签值。
            return_attns: 为 True 时返回每层注意力矩阵（可视化/调试用）。
        """
        logits, loss, _, attns, _ = self._forward_impl(
            idx,
            targets=targets,
            ignore_index=ignore_index,
            return_attns=return_attns,
        )
        if return_attns:
            return logits, loss, attns
        return logits, loss

    @torch.no_grad()
    def forward_trace(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """前向一次上下文，返回 logits、最终隐藏状态与各层注意力矩阵。

        供 ``generate_stream`` 做可视化追踪使用；训练/评估仍走 ``forward``，
        以免热路径保留注意力张量（显存开销为 ``n_layer * B * H * T^2``）。

        参数:
            idx: ``(B, T)`` token id，通常只有一行（单条提示）。
        返回:
            ``(logits, hidden, attns)``，``hidden`` 为 ``(B, T, n_embd)``。
        """
        logits, _, hidden, attns, _ = self._forward_impl(idx, return_attns=True)
        return logits, hidden, attns

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[int, torch.Tensor]:
        """从末位置 logits 采样一个 token，返回 ``(token_id, 概率向量)``。

        参数:
            logits: ``(1, 1, vocab_size)`` 的末位置 logits。
            temperature: 温度；``<= 0`` 表示 greedy。
            top_k: 只保留概率最高的 k 个，``<= 0`` 表示关闭。
            top_p: nucleus 累积阈值，``>= 1`` 表示关闭。
        返回:
            选中的 token id 与生效分布（形状 ``(1, vocab_size)``；greedy 时为
            T=1 的 softmax）。
        """
        token_id, probs, _ = GPT._sample_with_trace(logits, temperature, top_k, top_p)
        return token_id, probs

    @staticmethod
    def _sample_with_trace(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        trace_candidates: int = 0,
    ) -> tuple[int, torch.Tensor, dict[str, Any] | None]:
        """采样下一个 token，并可选地记录完整采样管线。

        随机数用显式逆 CDF（一个 ``torch.rand`` 标量 + 累积和）而非
        ``torch.multinomial``，这样可视化器能准确展示随机数落在累积轴的哪个
        位置；两者采样分布相同。

        ``trace_candidates=0`` 关闭追踪（训练/评估热路径不为此付费）。开启时
        返回的追踪包含候选集以及各中间分布：温度缩放、top-k 截断、top-p
        （nucleus）截断与最终抽样。

        参数:
            logits: ``(1, 1, vocab_size)`` 末位置 logits。
            temperature: 温度；``<= 0`` 走 greedy（不消耗随机数）。
            top_k: top-k 截断阈值，``<= 0`` 关闭。
            top_p: nucleus 阈值，``>= 1`` 关闭。
            trace_candidates: 追踪里保留的候选 token 数，0 表示不构建追踪。
        返回:
            ``(token_id, final_probs, trace_or_None)``。
        """

        vocab_size = int(logits.size(-1))
        greedy = float(temperature) <= 0.0
        if greedy:
            # greedy 分支不消耗随机数，确保同 seed 下与采样路径的随机流互不干扰。
            probs = F.softmax(logits, dim=-1)
            token_id = int(torch.argmax(logits, dim=-1).item())
            trace = None
            if trace_candidates > 0:
                trace = GPT._sample_trace_payload(
                    logits=logits[0],
                    raw_probs=probs[0],
                    stage_probs=probs[0],
                    final_probs=probs[0],
                    chosen=token_id,
                    u=None,
                    mode="greedy",
                    temperature=0.0,
                    top_k=top_k,
                    top_p=top_p,
                    top_k_threshold=None,
                    nucleus_mass=1.0,
                    trace_candidates=trace_candidates,
                    filters_applied=False,
                )
            return token_id, probs, trace

        # 规则内核（minigpt.sampling）统一温度/top-k/nucleus/逆 CDF 的判定逻辑；
        # 数值计算仍用 torch 张量，保持 float32 精度与既有随机数流不变。
        from minigpt import sampling as _sampling

        scaled = logits / float(temperature)
        raw_probs = F.softmax(scaled, dim=-1)
        threshold_value = _sampling.top_k_threshold(scaled[0].tolist(), top_k)
        if threshold_value is not None:
            threshold = torch.tensor(threshold_value, device=logits.device, dtype=scaled.dtype)
            keep = scaled >= threshold
            stage_probs = F.softmax(scaled.masked_fill(~keep, float("-inf")), dim=-1)
            top_k_threshold = threshold_value
        else:
            stage_probs = raw_probs
            top_k_threshold = None
        final_probs = stage_probs
        nucleus_mass = 1.0
        if top_p < 1.0:
            keep_mask = _sampling.nucleus_keep(stage_probs[0].tolist(), top_p)
            keep_tensor = torch.tensor(keep_mask, device=logits.device, dtype=torch.bool)
            kept = stage_probs.masked_fill(~keep_tensor, 0.0)
            nucleus_mass = float(kept.sum().item())
            # 重归一化；clamp_min(1e-12) 防止 nucleus 全被截掉时除零。
            final_probs = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cumulative = torch.cumsum(final_probs[0], dim=-1)
        u = float(torch.rand((), device=logits.device).item())
        token_id = _sampling.inverse_cdf_token(cumulative.tolist(), u)
        # 累积和因浮点误差可能略小于 1.0，越界下标钳回词表范围。
        token_id = min(max(token_id, 0), vocab_size - 1)
        trace = None
        if trace_candidates > 0:
            trace = GPT._sample_trace_payload(
                logits=logits[0],
                raw_probs=raw_probs[0],
                stage_probs=stage_probs[0],
                final_probs=final_probs[0],
                chosen=token_id,
                u=u,
                mode="sample",
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                top_k_threshold=top_k_threshold,
                nucleus_mass=nucleus_mass,
                trace_candidates=trace_candidates,
                filters_applied=True,
            )
        return token_id, final_probs, trace

    @staticmethod
    def _sample_trace_payload(
        *,
        logits: torch.Tensor,
        raw_probs: torch.Tensor,
        stage_probs: torch.Tensor,
        final_probs: torch.Tensor,
        chosen: int,
        u: float | None,
        mode: str,
        temperature: float,
        top_k: int,
        top_p: float,
        top_k_threshold: float | None,
        nucleus_mass: float,
        trace_candidates: int,
        filters_applied: bool,
    ) -> dict[str, Any]:
        """构建单个生成 token 的 JSON 可序列化采样追踪。

        参数:
            logits: ``(vocab_size,)`` 原始 logits。
            raw_probs: 温度缩放后的概率（尚未 top-k/top-p 截断）。
            stage_probs: top-k 截断并重归一后的概率。
            final_probs: nucleus 截断并重归一后的最终概率。
            chosen: 实际选中的 token id。
            u: 逆 CDF 使用的随机数，greedy 时为 ``None``。
            mode: ``"greedy"`` 或 ``"sample"``。
            temperature/top_k/top_p: 本次生成用的参数（原样记录）。
            top_k_threshold: top-k 对应的 logit 阈值，未启用时为 ``None``。
            nucleus_mass: nucleus 截断前的保留概率质量（用于展示丢弃了多少）。
            trace_candidates: 候选列表长度上限，至少为 1。
            filters_applied: 是否真正应用过采样截断（greedy 为 False）。
        返回:
            含候选列表、分布熵、nucleus 分段与最终抽样信息的字典；字段名与
            :func:`minigpt.sampling.build_sample_trace` 逐一对齐，纯后端可直接
            替换比较。
        """

        vocab_size = int(raw_probs.numel())
        count = max(1, min(int(trace_candidates), vocab_size))
        _, top_indices = torch.topk(raw_probs, count)
        indices = [int(index) for index in top_indices.tolist()]
        if chosen not in indices:
            # nucleus 重归一后选中 token 可能跌出原始 top-N，必须始终包含它，
            # 否则前端看不到「被选中的是哪一段」。
            indices[-1] = chosen
        raw_list = raw_probs.tolist()
        stage_list = stage_probs.tolist()
        final_list = final_probs.tolist()
        logit_list = logits.tolist()
        # T=1 的 softmax：用于展示「仅仅改变温度」对分布形状的影响。
        t1_list = F.softmax(logits, dim=-1).tolist()
        cumulative = torch.cumsum(final_probs, dim=-1).tolist()
        candidates: list[dict[str, Any]] = []
        for token_id in indices:
            prob_final = float(final_list[token_id])
            candidates.append(
                {
                    "token_id": token_id,
                    "logit": round(float(logit_list[token_id]), 6),
                    "prob_t1": round(float(t1_list[token_id]), 6),
                    "prob_raw": round(float(raw_list[token_id]), 6),
                    "prob_top_k": round(float(stage_list[token_id]), 6),
                    "prob_final": round(prob_final, 6),
                    "in_top_k": bool(stage_list[token_id] > 0.0),
                    "in_top_p": bool(prob_final > 0.0),
                    "cum_low": round(
                        float(cumulative[token_id - 1]) if token_id > 0 else 0.0, 6
                    ),
                    "cum_high": round(float(cumulative[token_id]), 6),
                    "chosen": token_id == chosen,
                }
            )
        candidates.sort(key=lambda candidate: candidate["prob_raw"], reverse=True)
        # clamp_min(1e-12)：概率为 0 的 token 贡献 0 * log2(0) 应视为 0，
        # 但不能直接对 0 取对数（会得到 -inf 并污染求和）。
        safe = final_probs.clamp_min(1e-12)
        entropy = float(-(final_probs * safe.log2()).sum().item())
        # 随机数按 token id 顺序在累积轴上行走；nucleus 足够小时输出各分段，
        # 让 UI 能画出精确的累积轴（超过 64 段时前端改用概要展示）。
        keep_ids = torch.nonzero(final_probs > 0, as_tuple=False).flatten().tolist()
        segments: list[dict[str, Any]] = []
        if len(keep_ids) <= 64:
            segments = [
                {
                    "token_id": int(token_id),
                    "prob": round(float(final_list[token_id]), 6),
                    "cum_low": round(
                        float(cumulative[token_id - 1]) if token_id > 0 else 0.0, 6
                    ),
                    "cum_high": round(float(cumulative[token_id]), 6),
                    "chosen": int(token_id) == chosen,
                }
                for token_id in keep_ids
            ]
        return {
            "mode": mode,
            "filters_applied": bool(filters_applied),
            "temperature": round(float(temperature), 4),
            "top_k": int(top_k),
            "top_p": round(float(top_p), 4),
            "vocab_size": vocab_size,
            "trace_candidates": count,
            "top_k_threshold_logit": (
                None if top_k_threshold is None else round(float(top_k_threshold), 6)
            ),
            "nucleus_mass": round(float(nucleus_mass), 6),
            "nucleus_count": int((final_probs > 0).sum().item()),
            "nucleus_segments": segments,
            "entropy_bits": round(entropy, 6),
            "top1_prob": round(float(final_probs.max().item()), 6),
            "u": None if u is None else round(float(u), 6),
            "chosen_id": int(chosen),
            "chosen_prob": round(float(final_list[chosen]), 6),
            "chosen_cum_low": round(
                float(cumulative[chosen - 1]) if chosen > 0 else 0.0, 6
            ),
            "chosen_cum_high": round(float(cumulative[chosen]), 6),
            "candidates": candidates,
        }

    @staticmethod
    def _apply_generation_controls(
        logits: torch.Tensor,
        history: list[int],
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        """重复惩罚 + n-gram 封禁（在温度/top-k/top-p 之前作用于 logits）。

        参数:
            logits: ``(1, 1, vocab_size)`` 末位置 logits。
            history: 已生成的完整 token 序列（含 prompt）。
            repetition_penalty: HF 约定的重复惩罚系数，1.0 表示关闭。
            no_repeat_ngram_size: 禁止重复的 n-gram 长度，0 表示关闭。
        返回:
            处理后的 logits；有改动时返回克隆，未改动时返回原张量（调用方不应
            原地修改返回值）。
        """

        if repetition_penalty != 1.0 and history:
            # 只处理历史中出现过的 token（去重后一次性索引，避免逐 token 循环）。
            seen = torch.tensor(sorted(set(history)), dtype=torch.long, device=logits.device)
            scores = logits[0, seen]
            # HF 约定：正 logit 除以惩罚系数，负 logit 乘以惩罚系数。
            penalized = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
            logits = logits.clone()
            logits[0, seen] = penalized
        from minigpt import sampling as _sampling

        banned = _sampling.ngram_banned_tokens(history, int(no_repeat_ngram_size))
        if banned:
            # -inf 使其在 softmax 后严格为 0，等价于从候选集中删除。
            logits = logits.clone()
            logits[0, torch.tensor(sorted(banned), dtype=torch.long, device=logits.device)] = float("-inf")
        return logits

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        eos_id: int | None = None,
        trace_top_k: int = 0,
        trace_attention: bool = False,
        trace_sampling: bool = False,
        trace_candidates: int = 20,
        use_cache: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_sequences: list[list[int]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """每个生成的 token 产出一个字典（KV cache 增量解码）。

        基础字段：``step``、``token_id``、``prob``、``topk``、``finished``、
        ``context_len``。``trace_attention=True`` 时每项另带 ``attention`` 与
        ``hidden``（注意力追踪需要完整矩阵，会退回全窗口重算）。
        ``trace_sampling=True`` 时另带 ``sampling``（采样管线追踪）。

        采样控制：``repetition_penalty``（1.0=关闭）、``no_repeat_ngram_size``
        （0=关闭）、``stop_sequences``（token id 序列，命中即停止，且该序列不计入
        输出）。``use_cache=False`` 退回每步全窗口重算（参照路径）。

        参数:
            prompt_ids: 提示 token 序列，不能为空。
            max_new_tokens: 最多生成多少个 token（不含 prompt）。
            temperature: 温度；``<= 0`` 走 greedy。
            top_k: top-k 阈值，``<= 0`` 关闭。
            top_p: nucleus 阈值，``>= 1`` 关闭。
            eos_id: 结束符 id；``None`` 表示不因 EOS 停止。
            trace_top_k: 每个 token 附带的前 k 个候选概率，0 关闭。
            trace_attention: 是否输出注意力矩阵与最终隐状态。
            trace_sampling: 是否输出完整采样追踪。
            trace_candidates: 采样追踪中的候选数。
            use_cache: 是否使用 KV cache 增量解码。
            repetition_penalty: 重复惩罚系数。
            no_repeat_ngram_size: 禁止重复的 n-gram 长度。
            stop_sequences: 多组停止序列（token id 列表），任一命中即停止。
        产出:
            每步一个字典；``finished`` 为 True 后生成器结束。
        异常:
            ValueError: ``prompt_ids`` 为空。
        """

        if not prompt_ids:
            raise ValueError("prompt must not be empty")
        device = next(self.parameters()).device
        block_size = self.config.block_size
        history = list(prompt_ids)
        stops = [list(s) for s in (stop_sequences or []) if s]
        # 注意力追踪需要完整注意力矩阵，增量路径拿不到历史矩阵，故强制全窗口。
        incremental = bool(use_cache) and not trace_attention

        idx = torch.tensor([history], dtype=torch.long, device=device)
        past_kv = None
        cached_len = 0
        if incremental:
            # 预填充：缓存 prompt 去掉末 token 的尾窗（最多 block_size-1 个），
            # 末 token 在首步增量喂入——若把它也预填充，首步会重复计入该
            # token 且位置整体偏移（与全窗口路径 logits 不一致）。
            prefix = idx[:, :-1]
            prefill = prefix[:, -(block_size - 1) :] if block_size > 1 else idx[:, :0]
            if prefill.size(1) > 0:
                _, _, past_kv = self.forward_with_cache(prefill)
                cached_len = prefill.size(1)

        for step in range(1, max_new_tokens + 1):
            if incremental:
                if cached_len >= block_size:
                    # 滑窗：丢弃最旧缓存，重建最近 block_size-1 个 token
                    # （同样不含末 token，末 token 由本步增量喂入）。
                    recent = torch.tensor(
                        [history[-block_size : -1]], dtype=torch.long, device=device
                    )
                    _, _, past_kv = self.forward_with_cache(recent)
                    cached_len = recent.size(1)
                last_token = torch.tensor([[history[-1]]], dtype=torch.long, device=device)
                logits, _, past_kv = self.forward_with_cache(last_token, past_kv=past_kv)
                cached_len += 1
                hidden, attns = None, None
                context_len = cached_len
            else:
                idx_cond = idx[:, -block_size:]
                if trace_attention:
                    logits, hidden, attns = self.forward_trace(idx_cond)
                else:
                    logits, _ = self(idx_cond)
                    hidden, attns = None, None
                context_len = int(idx_cond.size(1))

            last_logits = self._apply_generation_controls(
                logits[:, -1, :], history, repetition_penalty, no_repeat_ngram_size
            )
            sampling: dict[str, Any] | None = None
            if trace_sampling:
                next_id, probs, sampling = self._sample_with_trace(
                    last_logits,
                    temperature,
                    top_k,
                    top_p,
                    trace_candidates=trace_candidates,
                )
            else:
                next_id, probs = self._sample_next(last_logits, temperature, top_k, top_p)

            history.append(next_id)
            # 停止序列只比较历史尾部：一旦尾部等于某个停止序列即结束，
            # 并标记该 token 不属于最终输出（generate 会跳过它）。
            stopped_by_sequence = any(
                len(history) >= len(s) and history[-len(s) :] == s for s in stops
            )
            finished = (eos_id is not None and next_id == eos_id) or stopped_by_sequence
            item: dict[str, Any] = {
                "step": step,
                "token_id": next_id,
                "prob": float(probs[0, next_id].item()),
                "finished": finished,
                "context_len": context_len,
            }
            if stopped_by_sequence:
                item["stop_sequence"] = True
            if trace_top_k and trace_top_k > 0:
                # 以生效分布（final_probs）取 top-k，而不是原始 logits，
                # 这样展示的是真实可抽样集合。
                values, top_indices = torch.topk(probs, min(trace_top_k, probs.size(-1)))
                item["topk"] = [
                    {"token_id": int(index), "prob": float(value)}
                    for value, index in zip(values[0].tolist(), top_indices[0].tolist())
                ]
            else:
                item["topk"] = []
            if trace_attention:
                item["attention"] = [attn[0].detach().cpu().tolist() for attn in attns]
                item["hidden"] = hidden[0, -1].detach().cpu().tolist()
            if sampling is not None:
                item["sampling"] = sampling
            yield item
            if finished:
                break
            if not incremental:
                idx = torch.cat(
                    [idx, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                    dim=1,
                )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        eos_id: int | None = None,
        use_cache: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_sequences: list[list[int]] | None = None,
    ) -> list[int]:
        """贪心收集 :meth:`generate_stream` 的结果，返回完整 token 序列。

        与直接循环 ``generate_stream`` 的区别仅在于丢弃 ``finished`` 那一步的
        token（EOS 或停止序列的最后一个 token 不进入输出）。

        参数: 同 :meth:`generate_stream`（无追踪参数）。
        返回:
            ``prompt_ids`` 加上所有生成的 token id。
        """
        # Empty-prompt guard lives in generate_stream (single source of truth).
        out = list(prompt_ids)
        for item in self.generate_stream(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            use_cache=use_cache,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            stop_sequences=stop_sequences,
        ):
            if item["finished"]:
                # 停止序列命中的 token 不属于输出；EOS 同样不追加。
                break
            out.append(item["token_id"])
        return out


__all__ = ["GPTConfig", "CausalSelfAttention", "MLP", "Block", "GPT"]
