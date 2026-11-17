"""``CharTokenizer`` 单元测试（unittest，纯标准库，无网络）。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minigpt.registry import TOKENIZERS
from minigpt.tokenization import CharTokenizer, SPECIALS, SPECIALS_IDS
from minigpt.tokenization.base import Tokenizer


class CharTokenizerTest(unittest.TestCase):
    """覆盖 CharTokenizerTest 相关行为。"""
    def test_specials_are_fixed_and_first(self) -> None:
        """验证 `test_specials_are_fixed_and_first` 对应的行为约定。"""
        tok = CharTokenizer.build("cab")
        self.assertEqual(SPECIALS, ("<unk>", "<eos>", "<pad>"))
        self.assertEqual(tok.specials, {"<unk>": 0, "<eos>": 1, "<pad>": 2})
        self.assertEqual(tok.unk_id, 0)
        self.assertEqual(tok.eos_id, 1)
        self.assertEqual(tok.pad_id, 2)
        self.assertEqual(tok.vocab_size, 3 + 3)

    def test_build_uses_sorted_unique_chars(self) -> None:
        """验证 构建 相关行为。"""
        tok = CharTokenizer.build("cabca")
        self.assertEqual(tok.chars, ["a", "b", "c"])
        self.assertEqual(tok.encode("abc"), [3, 4, 5])
        self.assertEqual(tok.encode("cab"), [5, 3, 4])
        self.assertEqual(len(tok), 6)

    def test_build_from_train_only_marks_unknown(self) -> None:
        """验证 构建、训练 相关行为。"""
        train = "abc"
        tok = CharTokenizer.build(train)
        # 训练集没出现过的字符 -> <unk>
        self.assertEqual(tok.encode("azc"), [3, 0, 5])
        self.assertEqual(tok.decode([3, 0, 5]), "ac")
        self.assertEqual(tok.decode([3, 0, 5], skip_special=False), "a<unk>c")

    def test_encode_decode_roundtrip(self) -> None:
        """验证 `test_encode_decode_roundtrip` 对应的行为约定。"""
        text = "Hello, 世界!\nLine two.\t"
        tok = CharTokenizer.build(text)
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)
        self.assertEqual(len(ids), len(text))

    def test_add_eos(self) -> None:
        """验证 EOS 相关行为。"""
        tok = CharTokenizer.build("abc")
        ids = tok.encode("cab", add_eos=True)
        self.assertEqual(ids[-1], tok.eos_id)
        self.assertEqual(tok.decode(ids), "cab")
        self.assertEqual(tok.decode(ids, skip_special=False), "cab<eos>")
        self.assertEqual(tok.encode(""), [])

    def test_state_dict_roundtrip_and_json_safe(self) -> None:
        """验证 `test_state_dict_roundtrip_and_json_safe` 对应的行为约定。"""
        tok = CharTokenizer.build("abcdefg")
        state = tok.state_dict()
        self.assertEqual(state["type"], "char")
        self.assertEqual(state["specials"], SPECIALS_IDS)
        encoded = json.dumps(state, ensure_ascii=False)
        restored = CharTokenizer.from_state(json.loads(encoded))
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.encode("bag"), tok.encode("bag"))
        self.assertEqual(restored.vocab_size, tok.vocab_size)

    def test_from_state_rejects_bad_state(self) -> None:
        """验证 `test_from_state_rejects_bad_state` 对应的行为约定。"""
        good = CharTokenizer.build("abc").state_dict()
        with self.assertRaises(ValueError):
            CharTokenizer.from_state({"type": "bpe", "version": 1, "chars": []})
        with self.assertRaises(ValueError):
            CharTokenizer.from_state({**good, "version": 99})
        with self.assertRaises(ValueError):
            CharTokenizer.from_state({"type": "char", "version": 1})
        bad_specials = {"<unk>": 0, "<eos>": 1}
        with self.assertRaises(ValueError):
            CharTokenizer.from_state({**good, "specials": bad_specials})
        with self.assertRaises(ValueError):
            CharTokenizer.from_state({**good, "chars": ["ab"]})

    def test_save_and_load_directory_and_file(self) -> None:
        """验证 `test_save_and_load_directory_and_file` 对应的行为约定。"""
        tok = CharTokenizer.build("xyz")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "data"
            path = tok.save(directory)
            self.assertEqual(path, directory / "tokenizer.json")
            self.assertTrue(path.is_file())
            loaded = CharTokenizer.load(directory)
            self.assertEqual(loaded.state_dict(), tok.state_dict())
            loaded_from_file = CharTokenizer.load(path)
            self.assertEqual(loaded_from_file.encode("xyz"), tok.encode("xyz"))

    def test_load_missing_raises(self) -> None:
        """验证 `test_load_missing_raises` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                CharTokenizer.load(Path(tmp) / "nope")

    def test_token_repr(self) -> None:
        """验证 token 相关行为。"""
        tok = CharTokenizer.build("a\n\t")
        self.assertEqual(tok.token_repr(tok.unk_id), "<unk>")
        self.assertEqual(tok.token_repr(tok.eos_id), "<eos>")
        self.assertEqual(tok.token_repr(tok.pad_id), "<pad>")
        self.assertEqual(tok.token_repr(tok.encode("a")[0]), "a")
        self.assertEqual(tok.token_repr(tok.encode("\n")[0]), "\\n")
        with self.assertRaises(ValueError):
            tok.token_repr(tok.vocab_size + 5)

    def test_permuted_specials_are_rejected(self) -> None:
        # F26：id 集合 {0,1,2} 不够，必须强制规范映射 unk=0/eos=1/pad=2。
        """验证 `test_permuted_specials_are_rejected` 对应的行为约定。"""
        permuted = {"<unk>": 1, "<eos>": 0, "<pad>": 2}
        with self.assertRaises(ValueError):
            CharTokenizer(["a", "b"], specials=permuted)
        rotated = {"<unk>": 2, "<eos>": 0, "<pad>": 1}
        with self.assertRaises(ValueError):
            CharTokenizer(["a", "b"], specials=rotated)
        good = CharTokenizer.build("abc").state_dict()
        with self.assertRaises(ValueError):
            CharTokenizer.from_state({**good, "specials": permuted})

    def test_canonical_specials_roundtrip_unchanged(self) -> None:
        """验证 `test_canonical_specials_roundtrip_unchanged` 对应的行为约定。"""
        tok = CharTokenizer(["a", "b"], specials={"<unk>": 0, "<eos>": 1, "<pad>": 2})
        self.assertEqual(tok.specials, SPECIALS_IDS)
        self.assertEqual(tok.encode("ab"), [3, 4])
        self.assertEqual(tok.decode([3, 4]), "ab")
        self.assertEqual(tok.token_repr(0), "<unk>")
        self.assertEqual(tok.token_repr(1), "<eos>")
        self.assertEqual(tok.token_repr(2), "<pad>")
        restored = CharTokenizer.from_state(tok.state_dict())
        self.assertEqual(restored.state_dict(), tok.state_dict())

    def test_decode_out_of_range_raises(self) -> None:
        """验证 `test_decode_out_of_range_raises` 对应的行为约定。"""
        tok = CharTokenizer.build("abc")
        with self.assertRaises(ValueError):
            tok.decode([tok.vocab_size + 1])

    def test_empty_vocab(self) -> None:
        """验证 词表 相关行为。"""
        tok = CharTokenizer.build("")
        self.assertEqual(tok.vocab_size, 3)
        self.assertEqual(tok.encode("anything"), [0] * 8)
        self.assertEqual(tok.decode([0] * 8), "")

    def test_bad_input_types(self) -> None:
        """验证 `test_bad_input_types` 对应的行为约定。"""
        tok = CharTokenizer.build("abc")
        with self.assertRaises(TypeError):
            tok.encode(123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CharTokenizer.build(None)  # type: ignore[arg-type]

    def test_unicode_vocab(self) -> None:
        """验证 词表 相关行为。"""
        text = "中文测试，abc"
        tok = CharTokenizer.build(text)
        self.assertEqual(sorted(tok.chars), sorted(set(text)))
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertGreaterEqual(tok.vocab_size, 10)

    def test_registered_as_char(self) -> None:
        """验证 `test_registered_as_char` 对应的行为约定。"""
        self.assertIn("char", TOKENIZERS.names())
        self.assertIs(TOKENIZERS.get("char"), CharTokenizer)
        with self.assertRaises(KeyError):
            TOKENIZERS.get("bpe")

    def test_protocol_compliance(self) -> None:
        """验证 `test_protocol_compliance` 对应的行为约定。"""
        tok = CharTokenizer.build("abc")
        self.assertIsInstance(tok, Tokenizer)


if __name__ == "__main__":
    unittest.main()
