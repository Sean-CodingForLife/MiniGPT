<#
.SYNOPSIS
    MiniGPT 一键上手（Windows PowerShell 7+ / Windows PowerShell 5.1）。

.DESCRIPTION
    幂等流程：.venv 已存在则复用，否则 python -m venv 创建；
    安装 requirements.txt 依赖；运行全量 unittest；打印启动面板命令。
    任何一步失败立即停止（$ErrorActionPreference = "Stop"）。

    依赖安装默认走 requirements.txt（torch==2.15.0）。
    # CPU-only 安装提示（Windows 无 CUDA 时可用官方 CPU 源，避免拉取 CUDA 版）：
    #   python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu

.PARAMETER WhatIf
    干跑：只打印将要执行的步骤，不创建环境、不安装依赖、不运行测试。

.EXAMPLE
    pwsh -NoProfile -File .\setup.ps1
    pwsh -NoProfile -File .\setup.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location -LiteralPath $root

$venvDir = Join-Path $root ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

function Invoke-SetupStep {
    param(
        [Parameter(Mandatory = $true)][string]$Target,
        [Parameter(Mandatory = $true)][string]$Action,
        [Parameter(Mandatory = $true)][scriptblock]$Body
    )
    if ($PSCmdlet.ShouldProcess($Target, $Action)) {
        & $Body
        if ($LASTEXITCODE -ne 0) {
            throw "步骤失败（退出码 $LASTEXITCODE）：$Action"
        }
    }
}

Write-Host "== MiniGPT 一键上手 ==" -ForegroundColor Cyan

# 1) 虚拟环境：存在则复用（幂等），否则创建。
if (Test-Path -LiteralPath $venvPython) {
    Write-Host "[1/4] 复用已有 .venv：$venvPython"
}
else {
    Invoke-SetupStep -Target ".venv" -Action "python -m venv .venv（创建虚拟环境）" -Body {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCmd) {
            throw "未找到 python；请先安装 Python 3.11–3.13 并加入 PATH。"
        }
        & python -m venv $venvDir
    }
}

# 2) 依赖安装。
# CPU-only 提示：无 CUDA 的 Windows 机器可改用官方 CPU 源安装 torch：
#   python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
Invoke-SetupStep -Target "requirements.txt" -Action "pip install -r requirements.txt（安装依赖）" -Body {
    & $venvPython -m pip install -r (Join-Path $root "requirements.txt")
}

# 3) 全量测试（unittest，约 45s）。
Invoke-SetupStep -Target "tests" -Action "python -m unittest discover -s tests -t . -q（全量测试）" -Body {
    & $venvPython -m unittest discover -s tests -t . -q
}

# 4) 启动面板命令。
Write-Host ""
Write-Host "== 完成！启动 Web 控制面板 ==" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe dashboard.py"
Write-Host "  浏览器打开 http://127.0.0.1:8000/"
Write-Host "  命令行对话：.\.venv\Scripts\python.exe chat.py --checkpoint runs/<run_id>/best.pt --interactive"