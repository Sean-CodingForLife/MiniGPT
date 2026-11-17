"""N11: setup.ps1 一键上手脚本——可解析、幂等、失败即停、支持 -WhatIf 干跑。

不实际重建环境：只做 PowerShell 语法解析检查与 -WhatIf 干跑。
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.ps1"
PWSH = shutil.which("pwsh")

def parse_check_command(path: Path) -> str:
    """构造一段只用 PowerShell 官方 Parser 做语法解析的脚本，不执行 setup.ps1 本体。"""
    literal = "'" + str(path).replace("'", "''") + "'"
    return (
        "$tokens = $null; $errors = $null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile({literal}, [ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_.Message }; exit 1 }; "
        "exit 0"
    )


@unittest.skipIf(PWSH is None, "pwsh (PowerShell 7+) not available")
class SetupScriptTests(unittest.TestCase):
    """守卫 setup.ps1 的语法正确性、干跑完整性以及失败即停/幂等设计。"""

    def test_script_exists(self) -> None:
        """一键上手脚本必须存在于仓库根目录，否则 README 的指引失效。"""
        self.assertTrue(SETUP.is_file(), "setup.ps1 missing at repository root")

    def test_script_parses_without_syntax_errors(self) -> None:
        """脚本必须能被 PowerShell 解析器完整解析，避免用户执行时才暴露语法错误。"""
        result = subprocess.run(
            [PWSH, "-NoProfile", "-Command", parse_check_command(SETUP)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_whatif_dry_run_succeeds_without_rebuilding_env(self) -> None:
        """-WhatIf 干跑须成功且打印 venv/依赖/测试/面板四阶段，保证用户能预演而不改环境。"""
        result = subprocess.run(
            [PWSH, "-NoProfile", "-File", str(SETUP), "-WhatIf"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        # 干跑必须覆盖四个阶段：venv / 依赖 / 测试 / 启动面板。
        self.assertIn(".venv", output)
        self.assertIn("requirements.txt", output)
        self.assertIn("unittest", output)
        self.assertIn("dashboard.py", output)

    def test_script_is_fail_fast_and_idempotent_by_design(self) -> None:
        """脚本文本需同时体现 Stop 错误策略、ShouldProcess 支持与 .venv 复用（幂等）。"""
        text = SETUP.read_text(encoding="utf-8")
        self.assertIn('$ErrorActionPreference = "Stop"', text)
        self.assertIn("SupportsShouldProcess", text)
        # 幂等：已有 .venv 必须复用而不是无条件重建。
        self.assertIn("Test-Path", text)
        self.assertIn("-m venv", text)
        self.assertIn("pip install -r requirements.txt", text.replace('"', ""))
        # CPU 源提示（注释）。
        self.assertIn("download.pytorch.org/whl/cpu", text)


if __name__ == "__main__":
    unittest.main()