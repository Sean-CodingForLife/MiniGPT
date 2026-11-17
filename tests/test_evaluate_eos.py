"""evaluate.py 的 eos_rate 口径护栏（阶段 0 新增）。

已确认缺陷：len(continuation) < max_new_tokens 一律记 stopped_early 并
当作 eos_rate——凡"没跑满"都算 EOS。正确口径：只有生成序列真的以 EOS
结束才计入 eos_rate；跑满 max_new_tokens 属于长度截断，不是 EOS。

用假 bundle/假数据源驱动真实 main()，不启动 PyTorch。三种样本：
* "early"：第一步就生成 EOS（真 EOS）；
* "short"：生成 2 个普通 token 后结束（非 EOS 的提前结束）；
* "long"：跑满 max_new_tokens（长度截断）。
正确 eos_rate = 1/3；缺陷实现会把三者都算成 EOS（=1.0）。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate  # noqa: E402

EOS = 1  # CharTokenizer 规范：<eos>=1
FILLER = ord("a")


class FakeBackend:
    """按问题首字符决定生成行为（prompt_ids 为 ord 映射，[3] 是问题首字符）。"""

    def generate_stream(self, model, prompt_ids, max_new_tokens, eos_id=None, **kwargs):
        question_char = chr(prompt_ids[3]) if len(prompt_ids) > 3 else "x"
        if question_char == "e":  # early：立即 EOS
            yield {"step": 1, "token_id": eos_id, "prob": 1.0, "finished": True, "topk": []}
            return
        count = 2 if question_char == "s" else max_new_tokens  # short / long
        for step in range(1, count + 1):
            yield {"step": step, "token_id": FILLER, "prob": 1.0, "finished": False, "topk": []}

    def eval_step(self, model, batch):
        return types.SimpleNamespace(loss=1.0, tokens=8)


class FakeSource:
    """覆盖 FakeSource 相关行为。"""
    def __init__(self, data_dir, tokenizer=None, block_size=128):
        self.tokenizer = tokenizer

    def eval_batches(self, **kwargs):
        return [{"x": [[3] * 8], "y": [[3] * 8]}]


class FakeModel:
    """覆盖 FakeModel 相关行为。"""
    def num_parameters(self) -> int:
        return 16


def make_bundle():
    config = types.SimpleNamespace(batch_size=1, block_size=64, data_dir="")
    tokenizer = types.SimpleNamespace(
        encode=lambda text, **kw: [ord(ch) for ch in text],
        decode=lambda ids, **kw: "".join(chr(i) for i in ids if i > 2),
        state_dict=lambda: {"fake": True},
        eos_id=EOS,
        vocab_size=128,
    )
    return types.SimpleNamespace(
        payload={"stage": "sft", "model_config": {}},
        config=config,
        backend=FakeBackend(),
        model=FakeModel(),
        tokenizer=tokenizer,
    )


class EvaluateEosTests(unittest.TestCase):
    """覆盖 EvaluateEosTests 相关行为。"""
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-eos-")
        self.root = Path(self._tmp.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        records = [
            {"q": "early", "a": "yes", "category": "c"},
            {"q": "short", "a": "no", "category": "c"},
            {"q": "long", "a": "no", "category": "c"},
        ]
        for split in ("train", "val", "test"):
            (self.data_dir / f"{split}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                encoding="utf-8",
            )
        self.checkpoint = self.root / "fake.pt"
        self.checkpoint.write_bytes(b"fake")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_eos_rate_counts_only_true_eos(self) -> None:
        """验证 EOS 相关行为。"""
        bundle = make_bundle()
        out_dir = self.root / "eval_out"
        argv = [
            "--checkpoint", str(self.checkpoint),
            "--stage", "sft",
            "--data_dir", str(self.data_dir),
            "--split", "val",
            "--output_dir", str(out_dir),
            "--max_new_tokens", "4",
            "--num_batches", "1",
        ]
        with mock.patch.object(evaluate, "_load_bundle", return_value=bundle), \
             mock.patch.object(evaluate, "SFTJsonlSource", FakeSource), \
             contextlib.redirect_stdout(io.StringIO()):
            code = evaluate.main(argv)
        self.assertEqual(code, 0)
        report = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
        metrics = report.get("metrics", {})
        rate = report.get("eos_rate", metrics.get("eos_rate"))
        # 三条样本只有一条真 EOS：eos_rate 必须是 1/3，而不是 1.0。
        self.assertIsNotNone(rate, "评估报告必须包含 eos_rate")
        self.assertAlmostEqual(rate, 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()