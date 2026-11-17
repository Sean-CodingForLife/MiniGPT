"""N10: scripts/build_dist.py — dist 副本同步与 --check 漂移检测。

迷你 dist 在临时目录构造：验证同步只刷新白名单源码/模板/requirements，
权重与模型卡不动，MANIFEST 中源码条目哈希重算、其余条目保留；
--check 两态（一致退出码 0；漂移退出码 1 并列出漂移文件）。
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("build_dist", ROOT / "scripts" / "build_dist.py")
build_dist = importlib.util.module_from_spec(_SPEC)
sys.modules["build_dist"] = build_dist
_SPEC.loader.exec_module(build_dist)

from minigpt.bundle import RUNTIME_FILES, TEMPLATES  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_mini_dist(root: Path, name: str = "mini-bundle") -> Path:
    """按根源码构造一个迷你 dist 副本（权重/模型卡用占位文件）。"""
    bundle = root / "dist" / name
    for relative in RUNTIME_FILES:
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for template in TEMPLATES:
        shutil.copy2(ROOT / "bundle_templates" / template, bundle / template)
    shutil.copy2(ROOT / "requirements.txt", bundle / "requirements.txt")
    (bundle / "pytorch_model.bin").write_bytes(b"\x00fake-weights")
    (bundle / "weights.f32").write_bytes(b"\x00fake-pure-weights")
    (bundle / "MODEL_CARD.md").write_text("# placeholder card\n", encoding="utf-8")
    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[path.relative_to(bundle).as_posix()] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    manifest = {"format": "minigpt-bundle-v1", "format_version": 1, "files": files}
    (bundle / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


class BuildDistSyncTests(unittest.TestCase):
    """覆盖 BuildDistSyncTests 相关行为。"""
    def test_sync_refreshes_drifted_files_and_manifest(self) -> None:
        """验证 同步、清单 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_mini_dist(root)
            manifest_path = bundle / "MANIFEST.json"
            manifest_before = json.loads(manifest_path.read_text(encoding="utf-8"))
            card_before = (bundle / "MODEL_CARD.md").read_bytes()
            weights_before = (bundle / "pytorch_model.bin").read_bytes()

            # 制造漂移：两个白名单文件 + 模板 + requirements。
            drifted = ("minigpt/models/gpt.py", "minigpt/pure/gpt.py", "run.py", "requirements.txt")
            for relative in drifted:
                path = bundle / relative
                path.write_text(path.read_text(encoding="utf-8") + "\n# stale copy\n", encoding="utf-8")

            # 同步前 --check 必须能发现这些漂移。
            drift_before = build_dist.check_dist(root=ROOT, dist_dir=root / "dist")
            for relative in drifted:
                self.assertTrue(any(relative in item for item in drift_before), relative)

            changed = build_dist.sync_dist(root=ROOT, dist_dir=root / "dist")
            self.assertTrue(changed)
            for relative in drifted:
                source = ROOT / ("bundle_templates/" + relative if relative in TEMPLATES else relative)
                self.assertEqual(sha256(bundle / relative), sha256(source), relative)
            # 权重 / 模型卡保持原样。
            self.assertEqual((bundle / "MODEL_CARD.md").read_bytes(), card_before)
            self.assertEqual((bundle / "pytorch_model.bin").read_bytes(), weights_before)
            # MANIFEST：漂移条目哈希已重算，权重/模型卡条目不变，无新增条目。
            manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest_after["files"]), set(manifest_before["files"]))
            for relative in drifted:
                self.assertEqual(manifest_after["files"][relative]["sha256"], sha256(bundle / relative))
                self.assertEqual(manifest_after["files"][relative], manifest_before["files"][relative])
            for keep in ("pytorch_model.bin", "weights.f32", "MODEL_CARD.md"):
                self.assertEqual(manifest_after["files"][keep], manifest_before["files"][keep])
            # 同步后 --check 必须通过。
            self.assertEqual(build_dist.check_dist(root=ROOT, dist_dir=root / "dist"), [])

    def test_sync_is_idempotent(self) -> None:
        """验证 同步 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_mini_dist(root)
            self.assertEqual(build_dist.sync_dist(root=ROOT, dist_dir=root / "dist"), [])
            manifest = (root / "dist" / "mini-bundle" / "MANIFEST.json").read_bytes()
            self.assertEqual(build_dist.sync_dist(root=ROOT, dist_dir=root / "dist"), [])
            self.assertEqual((root / "dist" / "mini-bundle" / "MANIFEST.json").read_bytes(), manifest)


class BuildDistCheckTests(unittest.TestCase):
    """覆盖 BuildDistCheckTests 相关行为。"""
    def test_check_passes_on_synced_dist(self) -> None:
        """验证 dist 目录 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_mini_dist(root)
            self.assertEqual(build_dist.check_dist(root=ROOT, dist_dir=root / "dist"), [])

    def test_check_lists_drift_and_missing(self) -> None:
        """验证 漂移检测 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_mini_dist(root)
            (bundle / "minigpt" / "models" / "gpt.py").write_text("# drifted\n", encoding="utf-8")
            (bundle / "serve.py").unlink()
            drift = build_dist.check_dist(root=ROOT, dist_dir=root / "dist")
            joined = "\n".join(drift)
            self.assertIn("minigpt/models/gpt.py", joined)
            self.assertIn("serve.py", joined)
            self.assertIn("mini-bundle", joined)

    def test_main_check_exit_codes(self) -> None:
        """验证 `test_main_check_exit_codes` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_mini_dist(root)
            dist_dir = root / "dist"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = build_dist.main(["--check", "--root", str(ROOT), "--dist-dir", str(dist_dir)])
            self.assertEqual(code, 0)
            (dist_dir / "mini-bundle" / "run.py").write_text("# drifted\n", encoding="utf-8")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                code = build_dist.main(["--check", "--root", str(ROOT), "--dist-dir", str(dist_dir)])
            self.assertEqual(code, 1)
            self.assertIn("run.py", buffer.getvalue())

    def test_main_sync_then_check(self) -> None:
        """验证 同步 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_mini_dist(root)
            (bundle / "requirements.txt").write_text("torch==0.0.0\n", encoding="utf-8")
            dist_dir = root / "dist"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(build_dist.main(["--root", str(ROOT), "--dist-dir", str(dist_dir)]), 0)
                self.assertEqual(build_dist.main(["--check", "--root", str(ROOT), "--dist-dir", str(dist_dir)]), 0)
            self.assertEqual(sha256(bundle / "requirements.txt"), sha256(ROOT / "requirements.txt"))


if __name__ == "__main__":
    unittest.main()