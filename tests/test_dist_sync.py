"""F09: dist 漂移守卫——真实 dist 三副本必须与根源码逐文件 sha256 一致。

白名单源码（RUNTIME_FILES）+ run.py/serve.py 模板 + requirements.txt 由
``scripts/build_dist.py`` 同步；本守卫调用其 ``check_dist`` 函数，任何漂移
（含 MANIFEST 哈希过期）都会让测试失败。修复方式：运行
``python scripts/build_dist.py`` 后重跑。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("build_dist", ROOT / "scripts" / "build_dist.py")
build_dist = importlib.util.module_from_spec(_SPEC)
sys.modules["build_dist"] = build_dist
_SPEC.loader.exec_module(build_dist)

EXPECTED_BUNDLES = ("minigpt-en-sft", "minigpt-zh-pretrain", "minigpt-zh-sft")


class RealDistSyncGuardTests(unittest.TestCase):
    """确保 dist 三个发布包存在且与根源码逐文件哈希一致，防止发布物过期。"""

    def test_all_expected_bundles_exist(self) -> None:
        """三个语言/阶段组合的发布包目录都必须带 MANIFEST.json 才能被识别。"""
        found = {path.name for path in build_dist.discover_bundles(ROOT / "dist")}
        for name in EXPECTED_BUNDLES:
            self.assertIn(name, found, f"dist/{name} 缺失或没有 MANIFEST.json")

    def test_real_dist_matches_root_source(self) -> None:
        """真实 dist 副本与根目录白名单源码必须零漂移，否则说明忘记重新构建发布包。"""
        drift = build_dist.check_dist(root=ROOT, dist_dir=ROOT / "dist")
        self.assertEqual(
            drift,
            [],
            "dist 与根源码漂移；请运行 python scripts/build_dist.py 同步：\n" + "\n".join(drift),
        )


if __name__ == "__main__":
    unittest.main()