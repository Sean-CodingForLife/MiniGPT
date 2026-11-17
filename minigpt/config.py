"""配置对象、预设与优先级规则。

优先级（从低到高）::

    dataclass defaults < --preset < --config JSON < MINIGPT_* env < CLI flags

所有路径都相对项目根解析（除非本身是绝对路径）。项目根由本文件位置自动推导，
可用 ``MINIGPT_ROOT``（或 ``--project_root``）覆盖；这正是本项目能在不同机器、
不同工作目录下直接跑起来的原因。

架构位置：``Config`` 是所有 CLI、训练器、web 后端共享的唯一配置结构（扁平
dataclass），模型超参、数据准备、优化、采样与部署参数都放在这里，避免各处
自定义配置对象。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

PATH_FIELDS = {
    "data_dir",
    "runs_dir",
    "web_dir",
    "download_dir",
    "pretrain_data_dir",
    "tokenizer_dir",
    "init_from",
    "sft_input",
}
"""需要按项目根做相对/绝对转换的字段名集合（``resolve_paths`` 与可移植导出共用）。"""


def default_project_root() -> Path:
    """返回项目根目录：优先 ``MINIGPT_ROOT``，否则取本文件的上上级目录。

    返回:
        绝对路径；不检查目录是否存在。
    """
    env = os.environ.get("MINIGPT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


@dataclass
class Config:
    """所有 CLI 共享的扁平配置（未知字段会被拒绝）。

    字段按用途分组（运行时/模型/数据准备/优化/SFT/采样/部署/模式），每个字段
    都对应一个同名的 ``--flag`` 与 ``MINIGPT_<FIELD>`` 环境变量。
    典型用法::

        config = get_preset("debug_en")   # 或 parse_config(stage, argv)
        config.resolve_paths()            # 相对路径 -> 绝对路径
        config.validate()                 # 交叉校验取值范围
    """

    # runtime / paths
    project_root: str = ""
    data_dir: str = "data/shakespeare_en"
    runs_dir: str = "runs"
    web_dir: str = "web"
    download_dir: str = "data/raw"
    run_name: str = ""
    # 血缘：所属项目（成员权威仍在 projects/<id>/project.json）与上游 run。
    # parent_run_id 由服务端从 init_from 推导，客户端直传会被覆盖。
    project_id: str = ""
    parent_run_id: str = ""
    device: str = "auto"
    dtype: str = "auto"
    num_threads: int = 0
    seed: int = 1337
    eval_seed: int = 1234

    # model
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True
    backend_type: str = "torch"
    model_type: str = "gpt"
    tokenizer_type: str = "char"
    callbacks: str = "telemetry,sample"
    plugins: str = ""

    # data preparation
    dataset: str = "shakespeare_en"
    input_path: str = ""
    # 可选的逗号分隔 UTF-8/JSONL 训练文本，**只**用于构建词表（绝不用验证/测试集）。
    # 适合「预训练语料 + SFT prompt」联合建表（例如繁体语料 + 简体 SFT 提示）。
    # 路径按项目根解析。
    vocab_extra: str = ""
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    source_url: str = ""
    zh_books: str = "23950,23962,24264,23863,23839,7337"
    sft_input: str = ""
    sft_source: str = ""
    max_samples: int = 800
    tokenizer_dir: str = ""

    # optimization
    stage: str = "pretrain"
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 3000
    lr: float = 1e-3
    min_lr: float = 6e-5
    warmup_steps: int = 100
    # 学习率调度：cosine（默认，旧行为）/ wsd（warmup-stable-decay）/ linear。
    lr_schedule: str = "cosine"
    # wsd 调度末段线性衰减占 warmup 之后步数的比例（0~1）。
    decay_fraction: float = 0.2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20
    save_interval: int = 500
    patience: int = 10
    min_delta: float = 0.0

    # SFT
    init_from: str = ""
    pretrain_data_dir: str = "data/shakespeare_en"
    pretrain_mix_ratio: float = 0.15

    # sampling / inspection
    sample_prompt: str = "ROMEO:"
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 1.0
    max_new_tokens: int = 200

    # web / deployment
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    max_jobs: int = 1

    # modes
    # resume 兼容 bool 与 str：False/True 为旧用法；"auto" 在 run 目录自动找
    # last.pt（pure 后端 last.mgckpt）；"true"/"false" 按布尔解析。
    resume: str | bool = False
    mini_run: bool = False
    gate: bool = False

    # ------------------------------------------------------------------
    @staticmethod
    def resume_enabled(value: Any) -> bool:
        """resume 取值是否表示要续训（True / "auto" 为真）。"""

        return resume_enabled(value)

    def apply_dict(self, values: dict[str, Any]) -> None:
        """用字典批量覆盖配置字段（未知字段交给 validate 拒绝）。"""
        allowed = {f.name for f in fields(self)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        for key, value in values.items():
            setattr(self, key, value)

    def to_dict(self, portable: bool = False) -> dict[str, Any]:
        """导出配置字典；portable=True 时去掉本机绝对路径。"""
        data = asdict(self)
        if portable:
            root = Path(self.project_root or default_project_root()).resolve()
            for key in PATH_FIELDS:
                value = data.get(key, "")
                if not value:
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = (root / path).resolve()
                try:
                    data[key] = str(path.relative_to(root))
                except ValueError:
                    data[key] = str(path)
        return data

    def resolve_paths(self) -> None:
        """把相对路径按项目根解析为绝对路径。"""
        root = Path(self.project_root).expanduser().resolve() if self.project_root else default_project_root()
        self.project_root = str(root)
        for key in PATH_FIELDS:
            value = getattr(self, key)
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = root / path
            setattr(self, key, str(path.resolve()))

    def validate(self) -> None:
        """校验字段取值与组合约束，非法时抛 ValueError。"""
        self.resume = normalize_resume(self.resume)
        if self.stage not in {"pretrain", "sft"}:
            raise ValueError(f"stage must be pretrain or sft, got {self.stage!r}")
        if self.backend_type not in {"torch", "pure"}:
            raise ValueError(f"unsupported backend_type {self.backend_type!r}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"device must be auto/cpu/cuda, got {self.device!r}")
        if self.dtype not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive")
        if self.max_steps <= 0 or self.eval_interval <= 0 or self.eval_iters <= 0:
            raise ValueError("max_steps/eval_interval/eval_iters must be positive")
        if self.save_interval <= 0 or self.log_interval <= 0:
            raise ValueError("save_interval/log_interval must be positive")
        if self.warmup_steps < 0 or self.patience < 0:
            raise ValueError("warmup_steps/patience must be >= 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.val_fraction < 1.0 or not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("split fractions must be in [0, 1)")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")
        if not 0.0 <= self.pretrain_mix_ratio <= 1.0:
            raise ValueError("pretrain_mix_ratio must be in [0, 1]")
        if self.lr <= 0 or self.min_lr < 0:
            raise ValueError("lr must be > 0 and min_lr >= 0")
        if self.min_lr > self.lr:
            raise ValueError("min_lr must be <= lr")
        if self.lr_schedule not in {"cosine", "wsd", "linear"}:
            raise ValueError(f"lr_schedule must be cosine/wsd/linear, got {self.lr_schedule!r}")
        if not 0.0 <= self.decay_fraction <= 1.0:
            raise ValueError("decay_fraction must be in [0, 1]")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0 (0 silently zeroes gradients in the torch backend)")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("beta1/beta2 must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.backend_type == "pure" and (self.bias or not self.tie_weights):
            raise ValueError("the pure backend requires bias=False and tie_weights=True")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature < 0 or self.top_k < 0 or self.max_new_tokens < 0:
            raise ValueError("temperature/top_k/max_new_tokens must be >= 0")
        if self.max_jobs <= 0 or self.web_port <= 0:
            raise ValueError("max_jobs and web_port must be positive")
        if self.run_name and (Path(self.run_name).name != self.run_name or ".." in self.run_name):
            raise ValueError("run_name must be a plain directory name")
        for field_name in ("project_id", "parent_run_id"):
            value = getattr(self, field_name)
            if value and (Path(value).name != value or ".." in value):
                raise ValueError(f"{field_name} must be a plain directory name")


def _preset(**overrides: Any) -> Config:
    """返回指定名称的预设配置（内部构造，便于复用）。"""
    return replace(Config(), **overrides)


PRESETS: dict[str, Config] = {
    "debug_en": _preset(
        data_dir="data/shakespeare_en",
        dataset="shakespeare_en",
        sample_prompt="ROMEO:",
    ),
    "debug_zh": _preset(
        data_dir="data/zh_gutenberg",
        dataset="zh_gutenberg",
        sample_prompt="三国演义：",
        max_steps=1000,
        eval_interval=100,
        warmup_steps=100,
    ),
    "en_10m": _preset(
        data_dir="data/shakespeare_en",
        dataset="shakespeare_en",
        sample_prompt="ROMEO:",
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        batch_size=32,
        max_steps=5000,
        lr=6e-4,
        min_lr=6e-5,
        warmup_steps=200,
        eval_interval=250,
        eval_iters=100,
        save_interval=500,
        patience=10,
    ),
    "zh_10m": _preset(
        data_dir="data/zh_gutenberg",
        dataset="zh_gutenberg",
        sample_prompt="三国演义：",
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        batch_size=32,
        max_steps=5000,
        lr=6e-4,
        min_lr=6e-5,
        warmup_steps=200,
        eval_interval=250,
        eval_iters=100,
        save_interval=500,
        patience=10,
    ),
    "sft_debug_en": _preset(
        stage="sft",
        data_dir="data/sft_demo_en",
        pretrain_data_dir="data/shakespeare_en",
        sample_prompt="",
        max_steps=300,
        lr=1e-4,
        min_lr=1e-5,
        warmup_steps=20,
        eval_interval=50,
        eval_iters=20,
        log_interval=10,
        save_interval=100,
        patience=5,
    ),
    "sft_debug_zh": _preset(
        stage="sft",
        data_dir="data/sft_demo_zh",
        pretrain_data_dir="data/zh_gutenberg",
        sample_prompt="",
        max_steps=300,
        lr=1e-4,
        min_lr=1e-5,
        warmup_steps=20,
        eval_interval=50,
        eval_iters=20,
        log_interval=10,
        save_interval=100,
        patience=5,
    ),
    "sft_10m": _preset(
        stage="sft",
        data_dir="data/sft_10m",
        pretrain_data_dir="data/shakespeare_en",
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        batch_size=16,
        max_steps=2000,
        lr=1e-4,
        min_lr=1e-5,
        warmup_steps=100,
        eval_interval=100,
        eval_iters=50,
        save_interval=250,
        patience=5,
    ),
    # Stage 2 纯后端的小配置：2 层/2 头/32 维、block 32、batch 4，
    # 在纯 Python 里几十秒就能跑完，用于验证完整训练闭环而不是追求效果。
    "pure_debug_en": _preset(
        data_dir="data/shakespeare_en",
        dataset="shakespeare_en",
        sample_prompt="ROMEO:",
        backend_type="pure",
        device="cpu",
        dtype="float32",
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=32,
        batch_size=4,
        max_steps=50,
        lr=3e-3,
        min_lr=1e-4,
        warmup_steps=5,
        eval_interval=10,
        eval_iters=5,
        log_interval=5,
        save_interval=25,
        patience=0,
        num_threads=1,
    ),
    "pure_debug_zh": _preset(
        data_dir="data/zh_gutenberg",
        dataset="zh_gutenberg",
        sample_prompt="三国演义：",
        backend_type="pure",
        device="cpu",
        dtype="float32",
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=32,
        batch_size=4,
        max_steps=50,
        lr=3e-3,
        min_lr=1e-4,
        warmup_steps=5,
        eval_interval=10,
        eval_iters=5,
        log_interval=5,
        save_interval=25,
        patience=0,
        num_threads=1,
    ),
    "pure_sft_debug_en": _preset(
        stage="sft",
        data_dir="data/sft_demo_en",
        pretrain_data_dir="data/shakespeare_en",
        backend_type="pure",
        device="cpu",
        dtype="float32",
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=32,
        batch_size=4,
        max_steps=20,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=3,
        eval_interval=5,
        eval_iters=3,
        log_interval=5,
        save_interval=10,
        patience=0,
        num_threads=1,
    ),
}


def get_preset(name: str) -> Config:
    """按名称获取预设配置副本。"""
    try:
        return replace(PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}") from exc


def _field_type(field) -> type:
    """返回字段的期望类型，用于环境变量/命令行解析。"""
    default = field.default
    if isinstance(default, bool):
        return bool
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    return str


def _parse_bool(value: str) -> bool:
    """把常见真/假字符串解析为 bool。"""
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def normalize_resume(value: Any) -> bool | str:
    """归一化 resume：bool 原样；"auto" 保留；"true"/"false" 等按布尔解析。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        return _parse_bool(text)
    raise ValueError(f"resume must be a bool or 'auto'/'true'/'false', got {value!r}")


def resume_enabled(value: Any) -> bool:
    """resume 是否表示“要续训”（True 与 "auto" 都为真，"false"/False 为假）。"""

    return bool(normalize_resume(value))


def _parse_resume_arg(value: str) -> bool | str:
    """解析 --resume 的取值（auto/true/false）。"""
    try:
        return normalize_resume(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def apply_env(config: Config) -> None:
    """应用 MINIGPT_* 环境变量覆盖。"""
    for field in fields(config):
        env_name = f"MINIGPT_{field.name.upper()}"
        if field.name == "project_root" and env_name not in os.environ:
            env_name = "MINIGPT_ROOT"  # friendly alias
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        kind = _field_type(field)
        value: Any
        if field.name == "resume":
            value = normalize_resume(raw)
        elif kind is bool:
            value = _parse_bool(raw)
        elif kind is int:
            value = int(raw)
        elif kind is float:
            value = float(raw)
        else:
            value = raw
        setattr(config, field.name, value)


def build_parser(stage: str) -> argparse.ArgumentParser:
    """按阶段构造 argparse 解析器。"""
    default_preset = "debug_en" if stage == "pretrain" else "sft_debug_en"
    parser = argparse.ArgumentParser(
        description=f"MiniGPT {stage} CLI (config precedence: defaults < config.json < env < CLI)"
    )
    parser.add_argument("--preset", default=default_preset, choices=sorted(PRESETS))
    parser.add_argument("--config", default=None, help="JSON config file; fields override the preset")
    parser.add_argument("--list_presets", action="store_true")
    for field in fields(Config):
        arg = "--" + field.name
        kind = _field_type(field)
        if field.name == "resume":
            # --resume（裸开关=True，旧用法不变）、--resume auto/true/false；
            # --no-resume 保留为 False 的显式写法。
            parser.add_argument(
                arg,
                nargs="?",
                const=True,
                default=argparse.SUPPRESS,
                type=_parse_resume_arg,
                metavar="{auto,true,false}",
            )
            parser.add_argument(
                "--no-resume", dest="resume", action="store_const", const=False, default=argparse.SUPPRESS
            )
        elif kind is bool:
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(arg, type=kind, default=argparse.SUPPRESS)
    return parser


def parse_config(stage: str, argv: list[str] | None = None) -> Config:
    """解析默认值/配置文件/环境变量/命令行并返回 Config。"""
    parser = build_parser(stage)
    args = parser.parse_args(argv)
    if args.list_presets:
        for name in sorted(PRESETS):
            print(name)
        raise SystemExit(0)

    config = get_preset(args.preset)
    config.stage = stage

    if args.config:
        path = Path(args.config).expanduser()
        if not path.is_absolute():
            # 与其余路径约定一致：相对路径按项目根解析，而不是当前工作目录。
            root = getattr(args, "project_root", None) or config.project_root or default_project_root()
            path = Path(root).expanduser().resolve() / path
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("config file must contain a JSON object")
        config.apply_dict(data)
        config.stage = stage

    apply_env(config)

    # CLI wins over config/env.  argparse used SUPPRESS, so only explicit flags
    # are present in vars(args).
    for key, value in vars(args).items():
        if key in {"preset", "config", "list_presets"}:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    config.stage = stage
    config.resolve_paths()
    config.validate()
    return config


def overlay_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    *,
    stage: str = "pretrain",
    preset: str | None = None,
    field_map: dict[str, str] | None = None,
) -> argparse.Namespace:
    """给任意入口解析器叠加统一配置优先级：preset < config.json < env < 显式 CLI。

    用于 chat/evaluate/observe/visualize/export/dashboard 这类原本裸 argparse
    的入口：保留它们各自的旗标与默认值，同时获得 ``--preset``/``--config`` 与
    ``MINIGPT_*`` 环境变量能力。

    * ``field_map``：入口旗标 dest -> Config 字段名（如 ``{"prompt": "sample_prompt"}``）；
      未列出的 dest 若与 Config 字段同名则自动映射。
    * 只有**显式**出现在命令行里的旗标才覆盖配置层；入口自己的默认值在没有
      配置层命中时保持不变（不破坏旧行为）。
    * 额外注入 ``--preset``/``--config``/``--project_root`` 三个旗标（若入口已有
      同名旗标则复用，不重复添加）。
    """

    field_map = dict(field_map or {})
    config_fields = {f.name for f in fields(Config)}

    def has_flag(name: str) -> bool:
        return any(name in action.option_strings for action in parser._actions)

    if not has_flag("--preset"):
        parser.add_argument("--preset", default=None, help="配置预设名（见 pretrain.py --list_presets）")
    if not has_flag("--config"):
        parser.add_argument("--config", default=None, help="JSON 配置文件；字段覆盖预设")
    if not has_flag("--project_root"):
        parser.add_argument("--project_root", default=None, help="项目根目录（相对路径按此解析）")

    # 第一遍：用 SUPPRESS 副本识别哪些旗标是显式给出的（不含默认值）。
    explicit_parser = argparse.ArgumentParser(add_help=False)
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        # 探测副本只用于识别"显式出现"，不承担必填校验：required 一律抑制，
        # 否则 --help / 仅给 --preset 的场景会在第一遍就误报必填旗标缺失。
        kwargs: dict[str, Any] = {"dest": action.dest, "default": argparse.SUPPRESS, "required": False}
        if isinstance(action, argparse._StoreTrueAction):
            kwargs["action"] = "store_true"
        elif isinstance(action, argparse._StoreFalseAction):
            kwargs["action"] = "store_false"
        elif isinstance(action, argparse._StoreConstAction):
            kwargs["action"] = "store_const"
            kwargs["const"] = action.const
        elif isinstance(action, argparse._AppendAction):
            kwargs["action"] = "append"
            if action.type is not None:
                kwargs["type"] = action.type
        elif isinstance(action, argparse.BooleanOptionalAction):
            # 布尔开关：副本里用裸 store_true/store_false 探测显式出现。
            try:
                explicit_parser.add_argument(*action.option_strings, action="store_true", **kwargs)
            except argparse.ArgumentError:
                pass
            continue
        else:
            if action.nargs is not None:
                kwargs["nargs"] = action.nargs
            if action.type is not None:
                kwargs["type"] = action.type
            if action.choices is not None:
                kwargs["choices"] = action.choices
        try:
            explicit_parser.add_argument(*action.option_strings, **kwargs)
        except argparse.ArgumentError:
            continue
    explicit, _ = explicit_parser.parse_known_args(argv)
    explicit_keys = set(vars(explicit))

    # 第二遍：正常解析（保留入口默认值与 --help）。
    args = parser.parse_args(argv)

    # 配置层：preset < config.json < env。没有任何配置来源时不改动入口默认值。
    preset_name = getattr(args, "preset", None) or preset
    config_path = getattr(args, "config", None)
    has_env = any(name.startswith("MINIGPT_") for name in os.environ)
    if preset_name or config_path or has_env:
        config = get_preset(preset_name) if preset_name else Config()
        config.stage = stage
        if config_path:
            path = Path(config_path).expanduser()
            if not path.is_absolute():
                root = getattr(args, "project_root", None) or config.project_root or default_project_root()
                path = Path(root).expanduser().resolve() / path
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("config file must contain a JSON object")
            config.apply_dict(data)
            config.stage = stage
        apply_env(config)

        # 把配置层写回 args：显式 CLI 旗标优先，其余按 field_map/同名映射。
        for action in parser._actions:
            dest = action.dest
            if dest in {"help", "preset", "config", "project_root"} or not action.option_strings:
                continue
            if dest in explicit_keys:
                continue
            field = field_map.get(dest, dest if dest in config_fields else None)
            if field is None or not hasattr(config, field):
                continue
            setattr(args, dest, getattr(config, field))
    return args


__all__ = [
    "Config",
    "PRESETS",
    "PATH_FIELDS",
    "default_project_root",
    "get_preset",
    "apply_env",
    "build_parser",
    "parse_config",
    "overlay_config",
    "normalize_resume",
    "resume_enabled",
]
