"""``configs/`` 目录的只增删改查（安全名字 + 校验 + 原子写）。

从 job_manager.py 拆出（阶段 4）。job_manager.py 仍 re-export ``ConfigStore``，
对外导入路径不变。本模块只依赖标准库与 job_records 基础层。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from minigpt.config import default_project_root
from minigpt.training.job_records import (
    ConfigError,
    JobNotFound,
    atomic_write_json,
    normalize_config,
    validate_config_name,
)


class ConfigStore:
    """``configs/`` 目录的只增删改查（安全名字 + 校验 + 原子写）。"""

    def __init__(self, configs_dir: str | Path | None = None, project_root: str | Path | None = None) -> None:
        """指定配置目录与仓库根，构造配置存储对象。"""
        root = Path(project_root or default_project_root()).expanduser().resolve()
        directory = Path(configs_dir) if configs_dir is not None else Path("configs")
        if not directory.is_absolute():
            directory = root / directory
        self.project_root = root
        self.configs_dir = directory.expanduser().resolve()
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        """返回安全的配置文件路径（已做白名单校验）。"""

        clean = validate_config_name(name)
        path = (self.configs_dir / f"{clean}.json").resolve()
        if path.parent != self.configs_dir:
            raise ConfigError("配置路径越界")
        return path

    def list(self) -> list[dict[str, Any]]:
        """列出全部配置（按名字排序）。"""

        items: list[dict[str, Any]] = []
        for path in sorted(self.configs_dir.glob("*.json")):
            if not path.is_file():
                continue
            try:
                clean = validate_config_name(path.stem)
            except ConfigError:
                continue  # 跳过非本工具产生的文件
            stat = path.stat()
            items.append(
                {
                    "name": clean,
                    "file": path.name,
                    "size": stat.st_size,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime)
                    .astimezone()
                    .isoformat(timespec="seconds"),
                }
            )
        return items

    def exists(self, name: str) -> bool:
        """判断配置是否存在。"""

        return self.path_for(name).is_file()

    def load(self, name: str) -> dict[str, Any]:
        """读取配置对象（原始 JSON），不存在抛 :class:`JobNotFound`。"""

        path = self.path_for(name)
        if not path.is_file():
            raise JobNotFound(f"配置不存在: {validate_config_name(name)}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"配置文件无法解析: {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件必须是 JSON 对象: {path.name}")
        return data

    def get(self, name: str) -> dict[str, Any]:
        """API 友好视图：``{"name": ..., "config": {...}}``。"""

        clean = validate_config_name(name)
        return {"name": clean, "config": self.load(clean)}

    def save(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        """校验配置后原子写入；配置无效时抛 :class:`ConfigError`。"""

        clean = validate_config_name(name)
        if not isinstance(config, dict):
            raise ConfigError("config 必须是 JSON 对象")
        # 只校验、不改写用户提交的内容（相对路径保持可移植）。
        normalize_config(config_data=config, preset=None, project_root=self.project_root)
        path = self.path_for(clean)
        atomic_write_json(path, config)
        return {"name": clean, "file": path.name, "config": config}

    def delete(self, name: str) -> dict[str, Any]:
        """删除配置；不存在抛 :class:`JobNotFound`。"""

        clean = validate_config_name(name)
        path = self.path_for(clean)
        if not path.is_file():
            raise JobNotFound(f"配置不存在: {clean}")
        path.unlink()
        return {"name": clean, "deleted": True}


__all__ = ["ConfigStore"]