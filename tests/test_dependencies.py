"""依赖策略守卫：除 torch 外只允许标准库与项目内模块，且纯后端不得触碰 torch。

教学仓库的价值在于「零额外依赖装得上」，因此把 import 白名单、requirements
的精确版本、lock 文件一致性、前端无外链、源码无本机绝对路径都固化成断言。
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STDLIB = set(sys.stdlib_module_names)
ALLOWED_THIRD_PARTY = {"torch"}
LOCAL_MODULES = {path.stem for path in ROOT.glob("*.py")} | {"tests"}


def iter_python_files():
    """遍历项目源码（排除虚拟环境、运行产物与生成缓存目录）。"""
    excluded_dirs = {".venv", "runs", "data", "__pycache__", "dist", "screenshots"}
    for path in ROOT.rglob("*.py"):
        if any(part in excluded_dirs for part in path.parts):
            continue
        # 运行/测试残留目录（torchinductor 缓存、tempfile 前缀目录等）不参与
        # 依赖策略扫描：它们不是项目源码，可能含任意第三方 import。
        if any(
            part.startswith(("torchinductor_", "tmp", "minigpt-b4-runid-"))
            for part in path.parts
        ):
            continue
        yield path


class DependencyPolicyTests(unittest.TestCase):
    """校验依赖面、版本锁定与源码可移植性约束。"""

    def test_imports_only_stdlib_or_torch(self) -> None:
        """全仓库只允许 import 标准库、torch、项目内模块，否则用户装不上环境。"""
        violations = []
        for path in iter_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in STDLIB or name in ALLOWED_THIRD_PARTY or name in LOCAL_MODULES or name in {"minigpt", "_bootstrap"}:
                        continue
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {name}")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_pure_backend_has_no_torch_dependency(self) -> None:
        """pure 后端必须只用标准库，连测试侧也不得引入 torch，否则降级路径失效。"""
        for path in sorted((ROOT / "minigpt" / "pure").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                if not names:
                    continue
                self.assertNotIn("torch", names, f"{path.relative_to(ROOT)}:{node.lineno} imports torch")

    def test_requirements_pins_exact_torch(self) -> None:
        """requirements.txt 只应有一行且为 torch==x.y.z，避免版本漂移导致不可复现。"""
        lines = []
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"^torch==\d+\.\d+\.\d+$")

    def test_lockfile_matches_requirements(self) -> None:
        """lock 文件必须存在，且其 torch 版本与 requirements 一致（允许 +cpu 本地后缀）。"""
        lock = ROOT / "requirements.lock.txt"
        self.assertTrue(lock.is_file(), "requirements.lock.txt missing")
        pinned = {}
        for line in lock.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                name, version = line.split("==", 1)
                pinned[name.lower()] = version
        req_line = next(
            line.strip()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
        name, version = req_line.split("==", 1)
        self.assertIn(name.lower(), pinned)
        self.assertEqual(pinned[name.lower()], version.split("+")[0])

    def test_dashboard_does_not_import_torch(self) -> None:
        """控制台入口与服务端模块不得 import torch，保证无 GPU 机器也能开面板。"""
        for relative in ("dashboard.py", "minigpt/web/server.py"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("import torch", text, relative)

    def test_frontend_has_no_external_runtime_urls(self) -> None:
        """前端只允许引用本地资源与 127.0.0.1/localhost，保证离线可用且不泄隐私。"""
        files = [ROOT / "web" / "index.html", ROOT / "web" / "app.js", ROOT / "web" / "style.css"]
        files.extend(sorted((ROOT / "web" / "js").glob("*.js")))
        for path in files:
            text = path.read_text(encoding="utf-8")
            urls = re.findall(r"https?://[^\s\"')<]+", text)
            external = [
                url
                for url in urls
                if not url.startswith(("http://127.0.0.1", "http://localhost"))
            ]
            self.assertEqual(external, [], f"{path.relative_to(ROOT)} has external URLs: {external}")

    def test_no_machine_specific_paths_in_code(self) -> None:
        """源码里不能写死本机绝对路径，否则换机器即报错；断言文件自身需排除。"""
        # 把路径模式拆成多段拼接，避免本测试文件命中自己的正则。
        bad_patterns = [
            "C:" + "\\\\Users" + "\\\\Administrator",
            "/" + "home/[A-Za-z0-9_]+/",
            "C:" + "/Users/",
        ]
        for path in iter_python_files():
            if path.name == "test_dependencies.py":
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in bad_patterns:
                self.assertIsNone(re.search(pattern, text), f"{path.relative_to(ROOT)} matches {pattern}")


if __name__ == "__main__":
    unittest.main()
