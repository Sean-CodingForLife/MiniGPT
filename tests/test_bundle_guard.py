"""导出发布包的输出目录安全守卫：危险路径必须在删除任何文件之前就被拒绝。

回归点：``export_bundle(force=True)`` 会先 ``rmtree`` 输出目录，若路径校验
放在删除之后，误传项目根/盘符根/家目录会直接抹掉用户数据。因此这里用
mock 断言 rmtree 从未被调用，并确认正常子目录只是因 checkpoint 缺失失败。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from minigpt.bundle import ROOT, export_bundle


class ExportGuardTests(unittest.TestCase):
    """覆盖各类危险输出路径被拒绝，且正常嵌套路径不被误杀。"""

    def _assert_refused_without_deletion(self, output: Path) -> None:
        """断言导出被 ValueError 拒绝，同时确认根本没有执行目录删除。"""
        with mock.patch("minigpt.bundle.shutil.rmtree") as rmtree:
            with self.assertRaises(ValueError):
                export_bundle("nonexistent-checkpoint.pt", output, create_zip=False, force=True)
            rmtree.assert_not_called()

    def test_refuses_project_root_even_with_force(self) -> None:
        """即使 force=True 也不允许把项目根目录当作输出目录清空。"""
        self._assert_refused_without_deletion(ROOT)

    def test_refuses_drive_root(self) -> None:
        """盘符根（如 C:\\）属于最高危路径，必须无条件拒绝。"""
        self._assert_refused_without_deletion(Path(ROOT.anchor))

    def test_refuses_home_directory(self) -> None:
        """用户家目录整体删除会波及桌面/文档等个人数据，必须拒绝。"""
        self._assert_refused_without_deletion(Path.home())

    def test_refuses_desktop_directory(self) -> None:
        """桌面目录是常见的误选目标，同样在拒绝名单内。"""
        desktop = Path.home() / "Desktop"
        if not desktop.is_dir():
            self.skipTest("no Desktop directory on this machine")
        self._assert_refused_without_deletion(desktop)

    def test_refuses_documents_directory(self) -> None:
        """文档目录也被列入保护范围，防止误删用户文件。"""
        documents = Path.home() / "Documents"
        if not documents.is_dir():
            self.skipTest("no Documents directory on this machine")
        self._assert_refused_without_deletion(documents)

    def test_refuses_ancestor_of_project_root(self) -> None:
        """项目根的上级目录包含仓库本身，删除等同毁库，必须拒绝。"""
        self._assert_refused_without_deletion(ROOT.parent)

    def test_normal_subdirectory_still_allowed(self) -> None:
        """守卫不能过宽：全新嵌套目录应放行，只因子目录里的 checkpoint 不存在而失败。"""
        # 走的是「文件缺失」分支而不是路径守卫分支，用异常类型区分两者。
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "bundle"
            with self.assertRaises(FileNotFoundError):
                export_bundle("nonexistent-checkpoint.pt", target, create_zip=False, force=True)


if __name__ == "__main__":
    unittest.main()