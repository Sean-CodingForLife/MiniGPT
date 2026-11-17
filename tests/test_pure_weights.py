"""无 torch 依赖的权重文件格式（format v1）读写与损坏检测测试。

纯 Python 后端离线读取权重时，任何头部/偏移/校验错误都会读到错误字节，
因此这里既验证正常往返，也逐一构造篡改、截断、错误 magic 与形状不匹配的坏文件。
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from minigpt.pure import weights


def sample_state() -> dict[str, tuple[tuple[int, ...], list[float]]]:
    """构造含 2 维矩阵与 1 维向量的最小权重集合，覆盖形状元数据的序列化。"""
    return {
        "transformer.wte.weight": ((2, 2), [1.0, -2.0, 3.5, 4.25]),
        "transformer.ln_f.weight": ((2,), [1.0, 0.5]),
    }


class WeightFileTests(unittest.TestCase):
    """验证权重文件保存/加载的往返一致性及各类损坏输入的拒绝行为。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "weights.f32"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_roundtrip(self) -> None:
        """保存再加载后格式/版本/dtype、元数据、张量形状与数值都必须原样保留。"""
        index = weights.save_weights(self.path, sample_state(), {"run_id": "demo", "stage": "sft"})
        self.assertEqual(index["format"], weights.FORMAT)
        self.assertEqual(index["version"], weights.VERSION)
        self.assertEqual(index["dtype"], "float32")
        state, loaded_index = weights.load_weights(self.path)
        self.assertEqual(loaded_index["metadata"], {"run_id": "demo", "stage": "sft"})
        self.assertEqual(sorted(state), sorted(sample_state()))
        self.assertEqual(state["transformer.wte.weight"][0], (2, 2))
        for name, (_, expected) in sample_state().items():
            for left, right in zip(state[name][1], expected):
                self.assertAlmostEqual(left, right, places=6)

    def test_offsets_are_contiguous(self) -> None:
        """张量数据区必须首尾相接无空洞，否则 mmap/顺序读取会错位。"""
        index = weights.save_weights(self.path, sample_state())
        entries = sorted(index["tensors"].items(), key=lambda item: item[1]["offset"])
        expected_offset = 0
        for name, entry in entries:
            self.assertEqual(entry["offset"], expected_offset)
            expected_offset += entry["bytes"]

    def test_detects_tampered_payload(self) -> None:
        """翻转载荷最后一个字节后校验和必须报错，而 verify=False 时仍可强制读取。"""
        weights.save_weights(self.path, sample_state())
        raw = bytearray(self.path.read_bytes())
        raw[-1] ^= 0xFF
        self.path.write_bytes(bytes(raw))
        with self.assertRaises(weights.WeightFileError):
            weights.load_weights(self.path)
        state, _ = weights.load_weights(self.path, verify=False)
        self.assertIn("transformer.wte.weight", state)

    def test_detects_truncated_and_bad_header(self) -> None:
        """文件过短或头部长度为荒谬值时需抛 WeightFileError，而不是越界崩溃。"""
        self.path.write_bytes(b"\x01\x00")
        with self.assertRaises(weights.WeightFileError):
            weights.load_weights(self.path)
        self.path.write_bytes(struct.pack("<I", 999) + b"{}")
        with self.assertRaises(weights.WeightFileError):
            weights.load_weights(self.path)

    def test_rejects_wrong_shape(self) -> None:
        """元素个数与 shape 不匹配时保存端就应拒绝，避免写出自相矛盾的文件。"""
        with self.assertRaises(ValueError):
            weights.save_weights(self.path, {"bad": ((3,), [1.0, 2.0])})

    def test_rejects_non_weight_json(self) -> None:
        """头部 JSON 不是本格式（format 字段不符）时不能被误当成权重文件加载。"""
        payload = json.dumps({"format": "something-else", "version": 1}).encode("utf-8")
        self.path.write_bytes(struct.pack("<I", len(payload)) + payload)
        with self.assertRaises(weights.WeightFileError):
            weights.load_weights(self.path)


if __name__ == "__main__":
    unittest.main()
