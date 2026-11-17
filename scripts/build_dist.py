"""N10: 从根源码同步刷新 dist/ 下的推理包副本（不碰权重/模型卡）。

同步范围（逐文件 sha256 比对，只复制有漂移的文件）：

* ``minigpt.bundle.RUNTIME_FILES`` 白名单内的最小源码子集；
* ``bundle_templates/run.py`` / ``serve.py`` 模板（落到包根）；
* ``requirements.txt``。

权重（``pytorch_model.bin`` / ``weights.f32``）、tokenizer/config、
``MODEL_CARD.md`` / ``README.md`` 一律不动；``MANIFEST.json`` 只重算
上述被同步文件的 sha256/字节数条目，其余条目与元信息保持原样。

用法::

    python scripts/build_dist.py            # 同步全部 dist 副本
    python scripts/build_dist.py --check    # 只校验一致性；漂移时退出码 1 并列出文件

可选 ``--root`` / ``--dist-dir`` 指向其他源码根与 dist 目录（测试用）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# 允许从任意 cwd 运行：把项目根加入 sys.path（脚本位于 <root>/scripts/）。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minigpt.bundle import RUNTIME_FILES, TEMPLATES  # noqa: E402


def _sha256(path: Path) -> str:
    """分块计算文件 sha256（每块 1 MiB），用于漂移检测与 MANIFEST 更新。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def synced_files() -> list[tuple[str, str]]:
    """(包内相对路径, 根源码相对路径) 列表：白名单源码 + 模板 + requirements。"""
    pairs = [(relative, relative) for relative in RUNTIME_FILES]
    pairs += [(template, f"bundle_templates/{template}") for template in TEMPLATES]
    pairs.append(("requirements.txt", "requirements.txt"))
    return pairs


def discover_bundles(dist_dir: Path) -> list[Path]:
    """dist/ 下含 MANIFEST.json 的子目录即一个推理包副本。"""
    if not dist_dir.is_dir():
        return []
    return sorted(path for path in dist_dir.iterdir() if path.is_dir() and (path / "MANIFEST.json").is_file())


def check_dist(root: Path | None = None, dist_dir: Path | None = None) -> list[str]:
    """逐文件 sha256 校验 dist 与根源码一致；返回漂移描述列表（空=一致）。"""
    root = Path(root) if root is not None else ROOT
    dist_dir = Path(dist_dir) if dist_dir is not None else root / "dist"
    bundles = discover_bundles(dist_dir)
    if not bundles:
        return [f"no bundle directories with MANIFEST.json under {dist_dir}"]
    drift: list[str] = []
    pairs = synced_files()
    for bundle in bundles:
        for relative, source_relative in pairs:
            source = root / source_relative
            target = bundle / relative
            if not source.is_file():
                drift.append(f"{bundle.name}: missing root source {source_relative}")
                continue
            if not target.is_file():
                drift.append(f"{bundle.name}: missing {relative}")
                continue
            if _sha256(source) != _sha256(target):
                drift.append(f"{bundle.name}: drift {relative}")
        manifest_path = bundle / "MANIFEST.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            drift.append(f"{bundle.name}: unreadable MANIFEST.json ({error})")
            continue
        files = manifest.get("files")
        if not isinstance(files, dict):
            drift.append(f"{bundle.name}: MANIFEST.json has no files mapping")
            continue
        for relative, _ in pairs:
            entry = files.get(relative)
            target = bundle / relative
            if not isinstance(entry, dict):
                drift.append(f"{bundle.name}: MANIFEST missing entry {relative}")
                continue
            if target.is_file() and entry.get("sha256") != _sha256(target):
                drift.append(f"{bundle.name}: MANIFEST stale hash {relative}")
    return drift


def sync_dist(root: Path | None = None, dist_dir: Path | None = None) -> list[str]:
    """把根源码同步进每个 dist 副本；返回实际更新的文件描述（空=已一致）。"""
    root = Path(root) if root is not None else ROOT
    dist_dir = Path(dist_dir) if dist_dir is not None else root / "dist"
    bundles = discover_bundles(dist_dir)
    if not bundles:
        raise FileNotFoundError(f"no bundle directories with MANIFEST.json under {dist_dir}")
    pairs = synced_files()
    changed: list[str] = []
    for bundle in bundles:
        manifest_path = bundle / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})
        manifest_dirty = False
        for relative, source_relative in pairs:
            source = root / source_relative
            if not source.is_file():
                raise FileNotFoundError(f"root source missing: {source}")
            target = bundle / relative
            # 只在内容不同时才覆盖，避免无意义地改动 dist 时间戳。
            if not target.is_file() or _sha256(source) != _sha256(target):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                changed.append(f"{bundle.name}: {relative}")
            entry = files.get(relative)
            digest = _sha256(target)
            size = target.stat().st_size
            if not isinstance(entry, dict) or entry.get("sha256") != digest or entry.get("bytes") != size:
                files[relative] = {"sha256": digest, "bytes": size}
                manifest_dirty = True
        if manifest_dirty:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            changed.append(f"{bundle.name}: MANIFEST.json")
    return changed


def main(argv: list[str] | None = None) -> int:
    """命令行入口：默认同步，``--check`` 只校验。

    返回:
        0 表示已一致或同步成功；1 表示 ``--check`` 发现漂移（便于 CI 直接失败）。
    """
    parser = argparse.ArgumentParser(description="Sync dist bundles from the repository source (N10)")
    parser.add_argument("--check", action="store_true", help="only verify dist matches root source; exit 1 on drift")
    parser.add_argument("--root", default=None, help="repository root (default: this script's project)")
    parser.add_argument("--dist-dir", default=None, help="dist directory (default: <root>/dist)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else ROOT
    dist_dir = Path(args.dist_dir).resolve() if args.dist_dir else root / "dist"
    if args.check:
        drift = check_dist(root=root, dist_dir=dist_dir)
        if drift:
            print("dist drift detected:", file=sys.stderr)
            for item in drift:
                print(f"  {item}", file=sys.stderr)
            return 1
        print(f"dist OK: {len(discover_bundles(dist_dir))} bundle(s) match root source")
        return 0
    changed = sync_dist(root=root, dist_dir=dist_dir)
    if changed:
        print(f"synced {len(changed)} file(s):")
        for item in changed:
            print(f"  {item}")
    else:
        print("dist already in sync; nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())