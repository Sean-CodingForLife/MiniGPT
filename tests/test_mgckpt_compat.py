""".mgckpt 精度与向后兼容护栏（阶段 0 新增）。

现状：pure 后端训练用 float64，保存时 struct.pack("<f") 截成 float32，
续训后权重与 AdamW m/v 精度漂移。目标格式 v2 按 float64 保存且旧 v1
（float32）文件必须继续可读。这里用自造的 v1 夹具（与旧写端逐字节同构）
验证兼容读取，用 float64 往返验证无损。
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from minigpt.pure.checkpoint import (
    FORMAT,
    load_pure_checkpoint,
    save_pure_checkpoint,
)


def make_payload() -> dict:
    """构造 float64 下有意义、float32 下会被截断的高精度数值样本。"""
    # 这些值在 float32 下会丢精度（尾数超出 24 位），float64 下必须逐位还原。
    precise = [0.12345678901234567, -1.0000000000000002, 3.141592653589793, 1e-12]
    return {
        "format_version": 1,
        "backend": "pure",
        "stage": "pretrain",
        "iter_num": 7,
        "best_val_loss": 1.25,
        "config": {"backend_type": "pure"},
        "model_config": {"block_size": 8, "vocab_size": 12},
        "tokenizer_state": {"version": 1},
        "model": {"wte.weight": ([2, 2], precise)},
        "optimizer": {"m": [precise], "v": [[value * 2 for value in precise]], "step": 7},
    }


def write_v1_fixture(path: Path, payload: dict) -> None:
    """按旧版写端的字节布局手工构造 v1 文件（float32 payload）。"""
    tensors = []
    for name, (shape, values) in payload["model"].items():
        tensors.append((f"model.{name}", shape, values))
    for index, values in enumerate(payload["optimizer"]["m"]):
        tensors.append((f"optimizer.m.{index}", [len(values)], values))
    for index, values in enumerate(payload["optimizer"]["v"]):
        tensors.append((f"optimizer.v.{index}", [len(values)], values))
    meta = {key: value for key, value in payload.items() if key not in {"model", "optimizer"}}
    meta_optimizer = {
        key: value for key, value in payload["optimizer"].items() if key not in {"m", "v"}
    }
    header = {
        "format": FORMAT,
        "version": 1,
        "format_version": 1,
        "payload": meta,
        "optimizer": meta_optimizer,
        "tensors": {},
    }
    body = bytearray()
    for name, shape, values in tensors:
        encoded = b"".join(struct.pack("<f", value) for value in values)
        header["tensors"][name] = {"shape": shape, "offset": len(body), "bytes": len(encoded)}
        body.extend(encoded)
    header["payload_sha256"] = hashlib.sha256(bytes(body)).hexdigest()
    encoded_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<I", len(encoded_header)) + encoded_header + bytes(body))


class MgckptCompatTests(unittest.TestCase):
    """验证 v2（float64）无损往返，且旧 v1（float32）文件仍可按截断精度读取。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-mgckpt-")
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_float64_roundtrip_lossless(self) -> None:
        """权重与优化器 m/v 用 float64 保存后必须逐位还原，避免续训精度漂移。"""
        payload = make_payload()
        path = self.root / "last.mgckpt"
        save_pure_checkpoint(path, payload)
        loaded = load_pure_checkpoint(path)
        self.assertEqual(loaded["model"]["wte.weight"][1], payload["model"]["wte.weight"][1])
        self.assertEqual(loaded["optimizer"]["m"], payload["optimizer"]["m"])
        self.assertEqual(loaded["optimizer"]["v"], payload["optimizer"]["v"])
        self.assertEqual(loaded["iter_num"], 7)

    def test_v1_float32_file_still_readable(self) -> None:
        """旧版文件不得因格式升级而被拒绝，读回值需等于 float32 截断后的结果。"""
        payload = make_payload()
        path = self.root / "legacy.mgckpt"
        write_v1_fixture(path, payload)
        loaded = load_pure_checkpoint(path)
        # v1 是有损格式：读回值应等于 float32 截断后的值，而不是报错。
        expected = [struct.unpack("<f", struct.pack("<f", value))[0] for value in payload["model"]["wte.weight"][1]]
        self.assertEqual(loaded["model"]["wte.weight"][1], expected)
        self.assertEqual(loaded["optimizer"]["step"], 7)
        self.assertEqual(loaded["backend"], "pure")


if __name__ == "__main__":
    unittest.main()