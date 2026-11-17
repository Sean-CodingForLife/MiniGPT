"""MiniGPT 控制面板启动入口（纯标准库，可在没有 PyTorch 的机器上运行）。

用法::

    python dashboard.py                          # 127.0.0.1:8000
    python dashboard.py --port 0                 # 随机端口（测试用）
    python dashboard.py --host 0.0.0.0 --port 8000
    python dashboard.py --runs-dir D:/runs --web-dir web --configs-dir configs

如果看板所在解释器没有 PyTorch、而训练解释器有（例如项目的 ``.venv``），
可以显式指定训练子进程的解释器::

    python dashboard.py --python .venv/Scripts/python.exe

配置优先级（与 CONSTRUCTION §6.1 一致）：默认值 < ``MINIGPT_*`` 环境变量 <
命令行参数。服务关闭时（Ctrl+C）会停止由本服务启动的活动训练任务。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (把项目根加入 sys.path，支持任意 cwd 启动)
from minigpt.config import Config, apply_env, default_project_root, overlay_config
from minigpt.web.server import create_server


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器；默认值取自 :class:`minigpt.config.Config`。"""

    defaults = Config()
    parser = argparse.ArgumentParser(
        description="MiniGPT Web 控制面板：启停训练、在线校验/保存配置、实时曲线（纯标准库，无需 PyTorch）"
    )
    parser.add_argument("--host", default=None, help=f"监听地址，默认 {defaults.web_host}")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"监听端口，默认 {defaults.web_port}；0 表示由系统分配随机端口",
    )
    parser.add_argument("--runs-dir", default=None, help=f"运行目录，默认 {defaults.runs_dir}")
    parser.add_argument("--web-dir", default=None, help=f"静态资源目录，默认 {defaults.web_dir}")
    parser.add_argument("--configs-dir", default=None, help="配置目录，默认 configs")
    parser.add_argument(
        "--project-root",
        default=None,
        help="项目根目录（默认从 _bootstrap.py 自动探测，可用 MINIGPT_ROOT 覆盖）",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help=f"最大活动任务数，默认 {defaults.max_jobs}",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="训练子进程使用的 Python 解释器，默认当前解释器（可用带 PyTorch 的虚拟环境）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印每个 HTTP 请求日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、启动服务并阻塞直到 Ctrl+C。"""

    args = overlay_config(
        build_parser(),
        argv,
        stage="pretrain",
        field_map={
            "host": "web_host",
            "port": "web_port",
            "runs_dir": "runs_dir",
            "web_dir": "web_dir",
            "max_jobs": "max_jobs",
        },
    )

    # 优先级：默认值 < MINIGPT_* 环境变量 < 命令行参数（见 CONSTRUCTION §6.1）。
    config = Config()
    apply_env(config)  # MINIGPT_* 环境变量

    root = Path(args.project_root or config.project_root or default_project_root())
    root = root.expanduser().resolve()
    host = args.host or config.web_host
    port = config.web_port if args.port is None else args.port
    # 0 是合法值，表示让操作系统分配空闲端口（测试和临时启动常用）。
    if port < 0 or port > 65535:
        print(f"非法端口: {port}（允许 0-65535，0 表示随机端口）", file=sys.stderr)
        return 2
    max_jobs = config.max_jobs if args.max_jobs is None else args.max_jobs
    if max_jobs < 1:
        print(f"非法 --max-jobs: {max_jobs}（必须 >= 1）", file=sys.stderr)
        return 2

    python_executable = None
    if args.python:
        # 相对路径按项目根解析，方便写 --python .venv/Scripts/python.exe。
        python_path = Path(args.python)
        if not python_path.is_absolute():
            python_path = (root / python_path).resolve()
        if not python_path.is_file():
            print(f"指定训练解释器不存在: {python_path}", file=sys.stderr)
            return 2
        python_executable = str(python_path)

    try:
        server = create_server(
            host=host,
            port=port,
            project_root=root,
            runs_dir=args.runs_dir or config.runs_dir,
            web_dir=args.web_dir or config.web_dir,
            configs_dir=args.configs_dir or "configs",
            max_jobs=max_jobs,
            python_executable=python_executable,
            verbose=args.verbose,
        )
    except OSError as exc:
        print(f"无法监听 {host}:{port}: {exc}", file=sys.stderr)
        return 1

    bound_host, bound_port = server.server_address[0], server.server_address[1]
    url = f"http://{bound_host}:{bound_port}/"
    print(f"MiniGPT 控制面板: {url}", flush=True)
    print(f"  项目根目录: {root}", flush=True)
    print(f"  运行目录:   {server.app.runs_dir}", flush=True)
    print(f"  配置目录:   {server.app.configs_dir}", flush=True)
    print(f"  静态资源:   {server.app.web_dir}", flush=True)
    print(f"  训练解释器: {python_executable or sys.executable}", flush=True)
    print("按 Ctrl+C 停止（会同时停止正在运行的训练任务）。", flush=True)

    try:
        # poll_interval 取 0.2s：既能让 Ctrl+C 及时响应，也不会空转占满一个核。
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n正在停止服务...", flush=True)
    finally:
        server.server_close()  # 内部会调用 app.close()，停止活动训练任务
    print("服务已停止。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
