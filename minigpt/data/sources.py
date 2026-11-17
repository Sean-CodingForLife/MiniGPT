"""批量数据源：给训练器提供统一的 ``train_batch`` / ``eval_batches`` 接口。

这是数据层与训练层之间的稳定契约（backend/trainer 只看到 Python 数字）：

* :class:`PretrainSource` —— 读取 ``prepare.py`` 产出的 ``train/val/test.bin``；
* :class:`SFTJsonlSource` —— 读取 ``prepare_sft.py`` 产出的 JSONL，按
  ``Q: {q}\\nA: {a}<eos>`` 编码，仅 answer+EOS 计 loss（其余 -100）；
* :class:`MixedSource` —— SFT 训练时按 ``pretrain_ratio`` 混入预训练 batch。

batch 结构统一为::

    {"x": [[token_id, ...], ...], "y": [[token_id 或 -100, ...], ...]}

两个方法的签名固定为::

    train_batch(batch_size, block_size, rng) -> batch
    eval_batches(split, batch_size, block_size, num_batches, seed) -> [batch, ...]

本模块只依赖标准库与 tokenizer 协议，不导入 torch/numpy。
"""

from __future__ import annotations

import array
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..tokenization import CharTokenizer
from .binary import read_json, read_uint16_array, read_jsonl
from .formatters import encode_sft_example

#: ignore_index 约定（PyTorch cross_entropy 默认值）。
IGNORE_INDEX = -100

SPLITS: tuple[str, ...] = ("train", "val", "test")
EVAL_SPLITS: tuple[str, ...] = ("val", "test")


def load_tokenizer(data_dir: str | Path) -> CharTokenizer:
    """从数据目录（或 tokenizer.json 文件）加载 tokenizer。"""
    path = Path(data_dir)
    if path.is_dir():
        path = path / "tokenizer.json"
    return CharTokenizer.load(path)


# ----------------------------------------------------------------------
# 预训练数据源
# ----------------------------------------------------------------------
class PretrainSource:
    """字符级预训练数据源（读 ``<data_dir>/train.bin`` 等 + tokenizer/meta）。

    参数
    ----
    data_dir:
        ``prepare.py`` 的输出目录。
    tokenizer:
        可选注入；默认读取 ``<data_dir>/tokenizer.json``。
    """

    def __init__(self, data_dir: str | Path, tokenizer: Any | None = None):
        """校验数据目录结构并预加载训练集；val/test 在首次评估时才读入。"""
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"预训练数据目录不存在：{self.data_dir}；请先运行 prepare.py。"
            )
        self.meta_path = self.data_dir / "meta.json"
        self.metadata: dict[str, Any] = read_json(self.meta_path) if self.meta_path.is_file() else {}
        report_path = self.data_dir / "split_report.json"
        self.split_report: dict[str, Any] = read_json(report_path) if report_path.is_file() else {}
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(self.data_dir)
        self._validate_meta()
        # 延迟加载：验证集/测试集只在第一次评估时读入。
        self._ids: dict[str, array.array] = {}
        for split in SPLITS:
            path = self._split_path(split)
            if not path.is_file():
                raise FileNotFoundError(f"缺少预训练数据文件：{path}；请重新运行 prepare.py。")
        self._ids["train"] = read_uint16_array(self._split_path("train"))
        self._rng: Any | None = None

    # -- 元信息 --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """词表大小（由 tokenizer 决定，供后端建模型时使用）。"""
        return int(self.tokenizer.vocab_size)

    @property
    def data_meta(self) -> dict[str, Any]:
        """训练器写 checkpoint 的 ``data_meta`` 字段。"""
        return self.metadata

    def split_tokens(self, split: str) -> int:
        """某个切分的 token 数量（无需加载全部数据即可查询）。"""
        split = self._check_split(split)
        if split in self._ids:
            return len(self._ids[split])
        entry = (self.metadata.get("splits") or {}).get(split) or {}
        try:
            return int(entry.get("tokens", 0))
        except (TypeError, ValueError):
            return 0

    # -- 内部 ----------------------------------------------------------
    def _split_path(self, split: str) -> Path:
        """返回某个切分对应的 ``<split>.bin`` 路径。"""
        return self.data_dir / f"{split}.bin"

    def _check_split(self, split: str) -> str:
        """校验切分名合法（train/val/test），返回原名以便链式使用。"""
        if split not in SPLITS:
            raise ValueError(f"未知切分 {split!r}；可选：{SPLITS}")
        return split

    def _validate_meta(self) -> None:
        """核对 meta.json 与 tokenizer 的 vocab_size，防止混用数据目录。"""
        expected = self.metadata.get("vocab_size")
        if expected is not None and int(expected) != self.vocab_size:
            raise ValueError(
                f"meta.json 的 vocab_size={expected} 与 tokenizer.json 的 "
                f"{self.vocab_size} 不一致，数据目录可能被混用：{self.data_dir}"
            )

    def _load_split(self, split: str) -> array.array:
        """按需加载切分并缓存到 ``self._ids``（避免重复读盘）。"""
        split = self._check_split(split)
        if split not in self._ids:
            self._ids[split] = read_uint16_array(self._split_path(split))
        return self._ids[split]

    @staticmethod
    def _make_batch(ids: array.array, starts: Sequence[int], block_size: int) -> dict[str, list[list[int]]]:
        """按起点窗口切出 ``x``，并把标签设为右移一位的 ``y``（因果 LM）。"""
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        for start in starts:
            xs.append(list(ids[start:start + block_size]))
            # 预训练标签 = 下一个 token（因果语言模型）。
            ys.append(list(ids[start + 1:start + block_size + 1]))
        return {"x": xs, "y": ys}

    def _random_starts(self, ids: array.array, batch_size: int, block_size: int, rng: Any) -> list[int]:
        """随机采样 ``batch_size`` 个窗口起点；数据不足一个窗口时直接报错。"""
        if len(ids) < block_size + 1:
            raise ValueError(
                f"{self.data_dir} 的训练 token 太少（{len(ids)}），"
                f"至少需要 block_size + 1 = {block_size + 1}"
            )
        return [rng.randrange(0, len(ids) - block_size) for _ in range(batch_size)]

    # -- 对外接口 ------------------------------------------------------
    def train_batch(self, batch_size: int, block_size: int, rng: Any | None = None) -> dict[str, list[list[int]]]:
        """随机取一个预训练 batch；``y`` 为 ``x`` 向右平移一位。"""
        batch_size, block_size = int(batch_size), int(block_size)
        if batch_size <= 0 or block_size <= 1:
            raise ValueError("batch_size 必须 > 0 且 block_size 必须 >= 2")
        r = rng if rng is not None else (self._rng or random)
        ids = self._load_split("train")
        return self._make_batch(ids, self._random_starts(ids, batch_size, block_size, r), block_size)

    def eval_batches(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        num_batches: int,
        seed: Any = 1234,
    ) -> list[dict[str, list[list[int]]]]:
        """固定窗口的评估 batch：相同 seed 下每次得到完全相同的窗口。"""
        split = self._check_split(split)
        batch_size, block_size, num_batches = int(batch_size), int(block_size), int(num_batches)
        if batch_size <= 0 or block_size <= 1:
            raise ValueError("batch_size 必须 > 0 且 block_size 必须 >= 2")
        if num_batches <= 0:
            return []
        ids = self._load_split(split)
        rng = random.Random(f"eval:{seed}:{split}:{block_size}:{batch_size}")
        starts = self._random_starts(ids, batch_size * num_batches, block_size, rng)
        batches: list[dict[str, list[list[int]]]] = []
        for index in range(num_batches):
            chunk = starts[index * batch_size:(index + 1) * batch_size]
            batches.append(self._make_batch(ids, chunk, block_size))
        return batches


# ----------------------------------------------------------------------
# SFT 数据源
# ----------------------------------------------------------------------
class SFTJsonlSource:
    """SFT 数据源（读 ``<data_dir>/train.jsonl`` 等）。

    tokenizer 解析顺序：显式注入 -> ``<data_dir>/tokenizer.json`` ->
    ``sft_meta.json`` 记录的 ``tokenizer.tokenizer_dir`` -> 用 SFT 文本临时构建
    （最后一种只用于单测/兜底，正式流程应传预训练 tokenizer）。
    """

    def __init__(self, data_dir: str | Path, tokenizer: Any | None = None, block_size: int = 128):
        """加载 train/val/test 三个 JSONL，校验 q/a 字段并解析 tokenizer。"""
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"SFT 数据目录不存在：{self.data_dir}；请先运行 prepare_sft.py。"
            )
        meta_path = self.data_dir / "sft_meta.json"
        self.metadata: dict[str, Any] = read_json(meta_path) if meta_path.is_file() else {}
        report_path = self.data_dir / "split_report.json"
        self.split_report: dict[str, Any] = read_json(report_path) if report_path.is_file() else {}
        self.block_size = int(block_size)
        self._records: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            path = self.data_dir / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"缺少 SFT 数据文件：{path}；请重新运行 prepare_sft.py。")
            records = read_jsonl(path)
            for index, record in enumerate(records):
                if not isinstance(record.get("q"), str) or not isinstance(record.get("a"), str):
                    raise ValueError(f"{path} 第 {index + 1} 条缺少字符串字段 q/a")
            self._records[split] = records
        if not self._records["train"]:
            raise ValueError(f"SFT 训练集为空：{self.data_dir / 'train.jsonl'}")
        self.tokenizer = tokenizer if tokenizer is not None else self._resolve_tokenizer()
        self._rng: Any | None = None
        self._check_tokenizer_vocab()

    # -- tokenizer -----------------------------------------------------
    def _resolve_tokenizer(self) -> Any:
        """按「数据目录 -> sft_meta 记录目录 -> 临时构建」的顺序解析 tokenizer。"""
        candidates: list[Path] = [self.data_dir / "tokenizer.json"]
        tokenizer_info = self.metadata.get("tokenizer") or {}
        recorded = tokenizer_info.get("tokenizer_dir") or self.metadata.get("tokenizer_dir")
        if recorded:
            candidates.append(Path(recorded) / "tokenizer.json")
            candidates.append(Path(recorded))
        for path in candidates:
            if path.is_file():
                return CharTokenizer.load(path)
        print(
            f"[SFTJsonlSource] 警告：{self.data_dir} 没有 tokenizer.json，"
            "临时用 SFT 文本构建词表；正式训练请显式传入预训练 tokenizer。",
            file=sys.stderr,
        )
        text = "\n".join(
            f"Q: {record['q']}\nA: {record['a']}" for record in self._records["train"]
        )
        return CharTokenizer.build(text)

    def _check_tokenizer_vocab(self) -> None:
        """比对 sft_meta 记录的 vocab_size 与实际 tokenizer，不一致仅告警。

        这里不直接报错：历史数据可能记录了不同来源的 tokenizer，硬失败会
        让已有数据无法复现；真正的强校验在 Trainer 侧用同一 tokenizer 完成。
        """
        expected = (self.metadata.get("tokenizer") or {}).get("vocab_size")
        actual = getattr(self.tokenizer, "vocab_size", None)
        if expected is not None and actual is not None and int(expected) != int(actual):
            print(
                f"[SFTJsonlSource] 警告：sft_meta.json 记录的 vocab_size={expected} "
                f"与传入 tokenizer 的 {actual} 不一致。",
                file=sys.stderr,
            )

    # -- 元信息 --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """词表大小（由 tokenizer 决定，供后端建模型时使用）。"""
        return int(self.tokenizer.vocab_size)

    @property
    def data_meta(self) -> dict[str, Any]:
        """训练器写 checkpoint 的 ``data_meta`` 字段（SFT 数据的 meta）。"""
        return self.metadata

    @property
    def records(self) -> dict[str, list[dict[str, Any]]]:
        """按切分返回已加载的原始记录（只读用途：统计/调试）。"""
        return self._records

    def split_examples(self, split: str) -> int:
        """某个切分的样本条数。"""
        return len(self._records[self._check_split(split)])

    # -- 内部 ----------------------------------------------------------
    def _check_split(self, split: str) -> str:
        """校验切分名合法（train/val/test）。"""
        if split not in SPLITS:
            raise ValueError(f"未知切分 {split!r}；可选：{SPLITS}")
        return split

    def _encode(self, record: dict[str, Any], block_size: int) -> tuple[list[int], list[int]]:
        """把一条记录编码成 (x, y)，问题与 pad 位置在 y 中为 -100。"""
        return encode_sft_example(self.tokenizer, record["q"], record["a"], block_size)

    def _decode_batch(self, records: Iterable[dict[str, Any]], block_size: int) -> dict[str, list[list[int]]]:
        """把若干条记录编码并打包成统一 batch 结构。"""
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        for record in records:
            x, y = self._encode(record, block_size)
            xs.append(x)
            ys.append(y)
        return {"x": xs, "y": ys}

    # -- 对外接口 ------------------------------------------------------
    def train_batch(self, batch_size: int, block_size: int, rng: Any | None = None) -> dict[str, list[list[int]]]:
        """有放回随机采样 SFT 训练样本。"""
        batch_size, block_size = int(batch_size), int(block_size)
        if batch_size <= 0 or block_size <= 1:
            raise ValueError("batch_size 必须 > 0 且 block_size 必须 >= 2")
        r = rng if rng is not None else (self._rng or random)
        records = self._records["train"]
        sampled = [records[r.randrange(len(records))] for _ in range(batch_size)]
        return self._decode_batch(sampled, block_size)

    def eval_batches(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        num_batches: int,
        seed: Any = 1234,
    ) -> list[dict[str, list[list[int]]]]:
        """固定评估样本：相同 seed 下每次得到相同的样本集合。"""
        split = self._check_split(split)
        batch_size, block_size, num_batches = int(batch_size), int(block_size), int(num_batches)
        if batch_size <= 0 or block_size <= 1:
            raise ValueError("batch_size 必须 > 0 且 block_size 必须 >= 2")
        if num_batches <= 0:
            return []
        if split == "train":
            print("[SFTJsonlSource] 提示：eval_batches('train') 仅用于调试。", file=sys.stderr)
        records = self._records[split]
        # 先做一次确定性洗牌，再按顺序取不重叠窗口：批次之间不重复，
        # 避免「每个 batch 独立采样」带来的重复/偏置。
        rng = random.Random(f"eval:{seed}:{split}")
        order = list(range(len(records)))
        rng.shuffle(order)
        batches: list[dict[str, list[list[int]]]] = []
        cursor = 0
        for _ in range(num_batches):
            indices = [order[(cursor + j) % len(order)] for j in range(batch_size)]
            cursor = (cursor + batch_size) % len(order)
            sampled = [records[i] for i in indices]
            batches.append(self._decode_batch(sampled, block_size))
        return batches


# ----------------------------------------------------------------------
# 混合数据源（SFT + 预训练 replay）
# ----------------------------------------------------------------------
class MixedSource:
    """SFT 训练时按比例混入预训练 batch（CONSTRUCTION.md §5：默认 15%）。

    评估始终走 SFT 的 ``val/test``；``train_batch`` 按 ``pretrain_ratio`` 逐行
    混合两个数据源。若两边的 tokenizer 不一致，会统一改用预训练 tokenizer 并
    在 stderr 给出警告（模型词表以预训练 tokenizer 为准）。
    """

    def __init__(
        self,
        sft_source: SFTJsonlSource,
        pretrain_source: PretrainSource,
        pretrain_ratio: float = 0.15,
        rng: Any | None = None,
    ):
        """校验混合比例，并在两侧 tokenizer 不一致时统一到预训练 tokenizer。

        参数:
            sft_source: SFT 数据源（其 tokenizer 可能被就地替换）。
            pretrain_source: 预训练数据源，提供 replay 批次。
            pretrain_ratio: 每行样本来自预训练的概率，取值 [0, 1]。
            rng: 可选随机源；默认使用全局 ``random``。
        异常:
            ValueError: 比例越界。
        """
        ratio = float(pretrain_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"pretrain_ratio 必须在 [0, 1]，收到 {pretrain_ratio}")
        self.sft_source = sft_source
        self.pretrain_source = pretrain_source
        self.pretrain_ratio = ratio
        self._rng = rng
        pretrain_tokenizer = getattr(pretrain_source, "tokenizer", None)
        sft_tokenizer = getattr(sft_source, "tokenizer", None)
        if (
            pretrain_tokenizer is not None
            and sft_tokenizer is not None
            and sft_tokenizer is not pretrain_tokenizer
            and _tokenizer_state(sft_tokenizer) != _tokenizer_state(pretrain_tokenizer)
        ):
            print(
                "[MixedSource] 警告：SFT tokenizer 与预训练 tokenizer 不一致，"
                "已统一使用预训练 tokenizer。",
                file=sys.stderr,
            )
            self.sft_source.tokenizer = pretrain_tokenizer
        self.tokenizer = pretrain_tokenizer if pretrain_tokenizer is not None else sft_tokenizer

    # -- 元信息 --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """词表大小（由 tokenizer 决定，供后端建模型时使用）。"""
        return int(self.tokenizer.vocab_size)

    @property
    def data_meta(self) -> dict[str, Any]:
        """同时记录 SFT/预训练两边的 meta 与混合比例，写入 checkpoint。"""
        return {
            "pretrain": getattr(self.pretrain_source, "metadata", {}),
            "sft": getattr(self.sft_source, "metadata", {}),
            "pretrain_ratio": self.pretrain_ratio,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """``data_meta`` 的别名，兼容只认 ``metadata`` 的调用方。"""
        return self.data_meta

    # -- 对外接口 ------------------------------------------------------
    def train_batch(self, batch_size: int, block_size: int, rng: Any | None = None) -> dict[str, list[list[int]]]:
        """按 ``pretrain_ratio`` 混合 SFT 与预训练样本。"""
        batch_size = int(batch_size)
        block_size = int(block_size)
        r = rng if rng is not None else (self._rng or random)
        if self.pretrain_ratio <= 0.0:
            return self.sft_source.train_batch(batch_size, block_size, r)
        if self.pretrain_ratio >= 1.0:
            return self.pretrain_source.train_batch(batch_size, block_size, r)
        # 先逐行决定来源（每行一次 Bernoulli，随机性与逐行挑选完全一致），
        # 再按比例只抽所需行数，避免两侧各抽一个完整 batch 的约 2 倍开销。
        use_pretrain = [r.random() < self.pretrain_ratio for _ in range(batch_size)]
        n_pretrain = sum(use_pretrain)
        n_sft = batch_size - n_pretrain
        sft_batch = self.sft_source.train_batch(n_sft, block_size, r) if n_sft else None
        pretrain_batch = self.pretrain_source.train_batch(n_pretrain, block_size, r) if n_pretrain else None
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        sft_index = pretrain_index = 0
        for is_pretrain in use_pretrain:
            if is_pretrain:
                source, index = pretrain_batch, pretrain_index
                pretrain_index += 1
            else:
                source, index = sft_batch, sft_index
                sft_index += 1
            xs.append(list(source["x"][index]))
            ys.append(list(source["y"][index]))
        return {"x": xs, "y": ys}

    def eval_batches(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        num_batches: int,
        seed: Any = 1234,
    ) -> list[dict[str, list[list[int]]]]:
        """评估只使用 SFT 的 val/test（与训练混合比例无关）。"""
        return self.sft_source.eval_batches(split, batch_size, block_size, num_batches, seed)


def _tokenizer_state(tokenizer: Any) -> Any | None:
    """尽量取出 tokenizer 的可比较状态；取不到返回 None（视为不比较）。"""
    state_dict = getattr(tokenizer, "state_dict", None)
    if not callable(state_dict):
        return None
    try:
        return state_dict()
    except Exception:  # pragma: no cover - 自定义 tokenizer 可能未实现
        return None


__all__ = [
    "IGNORE_INDEX",
    "SPLITS",
    "EVAL_SPLITS",
    "load_tokenizer",
    "PretrainSource",
    "SFTJsonlSource",
    "MixedSource",
]
