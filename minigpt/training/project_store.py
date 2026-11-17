"""``projects/`` 目录的项目级管理（流水线成员关系的唯一权威）。

一个项目 = 数据 → 预训练 → SFT → 评测 → 导出 的全流程容器；
``project.json`` 的 ``run_ids`` 有序记录成员 run，挂载不改写 run 的冻结配置。
与 ConfigStore 不同：看板是 ThreadingHTTPServer，本存储自带锁保证并发安全。
只依赖标准库与 job_records 基础层。
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from minigpt.config import default_project_root
from minigpt.training.job_records import (
    ConfigError,
    JobNotFound,
    atomic_write_json,
    iso_now,
    validate_run_id,
)


class ProjectStore:
    """``projects/<id>/project.json`` 的增删查与成员挂载（线程安全、原子写）。"""

    def __init__(self, projects_dir: str | Path | None = None, project_root: str | Path | None = None) -> None:
        """指定项目状态目录与仓库根，构造存储对象。"""
        root = Path(project_root or default_project_root()).expanduser().resolve()
        directory = Path(projects_dir) if projects_dir is not None else Path("projects")
        if not directory.is_absolute():
            directory = root / directory
        self.project_root = root
        self.projects_dir = directory.expanduser().resolve()
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path_for(self, project_id: str) -> Path:
        """返回安全的项目文件路径（复用 run id 白名单校验）。"""

        clean = validate_run_id(project_id)
        path = (self.projects_dir / clean / "project.json").resolve()
        if path.parent.parent != self.projects_dir:
            raise ConfigError("项目路径越界")
        return path

    def exists(self, project_id: str) -> bool:
        """判断项目 id 是否已有 project.json。"""
        return self.path_for(project_id).is_file()

    def load(self, project_id: str) -> dict[str, Any]:
        """读取单个项目元数据；不存在或损坏时返回 None。"""
        path = self.path_for(project_id)
        if not path.is_file():
            raise JobNotFound(f"项目不存在: {validate_run_id(project_id)}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"项目文件无法解析: {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"项目文件必须是 JSON 对象: {path.name}")
        data.setdefault("run_ids", [])
        return data

    def list(self) -> list[dict[str, Any]]:
        """列出全部项目（按创建时间倒序）。"""

        items: list[dict[str, Any]] = []
        for directory in sorted(self.projects_dir.iterdir()):
            if not directory.is_dir():
                continue
            try:
                data = self.load(directory.name)
            except (ConfigError, JobNotFound):
                continue  # 跳过非本工具产生的目录
            items.append(
                {
                    "project_id": directory.name,
                    "name": data.get("name") or directory.name,
                    "note": data.get("note") or "",
                    "created_at": data.get("created_at"),
                    "run_ids": list(data.get("run_ids") or []),
                }
            )
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return items

    def create(self, name: str, note: str = "", project_id: str | None = None) -> dict[str, Any]:
        """创建新项目并落盘（id 缺省时自动生成）。"""
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("项目名称不能为空")
        if project_id is not None:
            clean = validate_run_id(project_id)
        else:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")[:48]
            clean = validate_run_id(f"{slug or 'project'}-{uuid.uuid4().hex[:6]}")
        with self._lock:
            path = self.path_for(clean)
            if path.is_file():
                raise ConfigError(f"项目已存在: {clean}")
            data = {
                "project_id": clean,
                "name": name.strip(),
                "note": str(note or ""),
                "created_at": iso_now(),
                "run_ids": [],
            }
            atomic_write_json(path, data)
            return data

    def attach_run(self, project_id: str, run_id: str) -> dict[str, Any]:
        """把已有 run 挂进项目（幂等、保序）；不改写 run 的冻结配置。"""

        clean_run = validate_run_id(run_id)
        with self._lock:
            data = self.load(project_id)
            runs = list(data.get("run_ids") or [])
            if clean_run not in runs:
                runs.append(clean_run)
            data["run_ids"] = runs
            atomic_write_json(self.path_for(project_id), data)
            return data

    def project_of(self, run_id: str) -> str | None:
        """反查 run 所属项目（成员权威只在 project.json，避免双写）。"""

        for item in self.list():
            if run_id in item["run_ids"]:
                return item["project_id"]
        return None


__all__ = ["ProjectStore"]
