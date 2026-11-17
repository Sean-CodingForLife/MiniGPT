"""语料下载与清洗（仅标准库；失败可重试、可本地兜底）。

数据源
------
* 英文：Tiny Shakespeare（与 nanoGPT/char-rnn 同一份 1,115,394 字节文件）。
  jsDelivr 为主源，raw.githubusercontent.com 为回退镜像；下载后用固定
  sha256 校验（``86c4e6aa...``）。
* 中文：Project Gutenberg 公版书（默认 6 本，书单可配置）。下载后去掉
  Gutenberg 页眉/页脚（``*** START/END OF ... ***``），逐本校验 START/END
  标记与长度，防止网络截断产生半本书。
* 公开 SFT：``shibing624/alpaca-zh`` 的 ``alpaca_gpt4_data_zh.json``（约 35MB，
  CC BY-NC 4.0，仅研究用途）；只缓存到 ``download_dir``，不提交仓库。

所有下载函数优先复用缓存；``--input``/``load_local_text`` 提供完全离线的
本地文本兜底。注意：本模块只负责“拿到干净文本”，分词与切分在
``prepare.py``。
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from .binary import atomic_write_bytes, atomic_write_text, read_json, sha256_bytes, sha256_file

#: 请求 UA：Gutenberg 会拒绝空 UA；带上项目标识便于对方识别（无外链依赖）。
USER_AGENT = "MiniGPT/1.0 (educational char-level LM; standard library only)"

#: Tiny Shakespeare 固定校验值（1,115,394 字节，65 个不同字符）。
SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
SHAKESPEARE_SIZE = 1_115_394
SHAKESPEARE_URLS: tuple[str, ...] = (
    "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
)

#: 默认中文书单（CONSTRUCTION.md §5）。
GUTENBERG_BOOK_IDS: tuple[str, ...] = ("23950", "23962", "24264", "23863", "23839", "7337")
GUTENBERG_URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_FALLBACK_TEMPLATES: tuple[str, ...] = (
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
)

#: 公开中文指令数据（只缓存不提交；CC BY-NC 4.0，仅研究）。
ALPACA_ZH_URL = "https://hf-mirror.com/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json"
ALPACA_ZH_FILENAME = "alpaca_gpt4_data_zh.json"
ALPACA_ZH_LICENSE = "CC BY-NC 4.0 (research only; not redistributed)"
#: 官方源完整性钉住（与 Shakespeare 同等待遇）；自定义 --source_url 不强制。
ALPACA_ZH_SHA256 = "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"

#: 单次下载字节上限：语料最大约 35MB，留 5 倍余量防异常响应耗尽内存/磁盘。
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024

#: Gutenberg 页眉/页脚标记（标题里可能含中文，故只匹配英文固定部分）。
_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*?\*\*\*", re.IGNORECASE
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*?\*\*\*", re.IGNORECASE
)
_END_LOOSE_RE = re.compile(r"^\s*End of (?:the )?Project Gutenberg.*$", re.IGNORECASE | re.MULTILINE)
_TITLE_RE = re.compile(r"^Title:\s*(.+?)\s*$", re.MULTILINE)

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),   # CJK 扩展 A
    (0x4E00, 0x9FFF),   # CJK 基本区
    (0xF900, 0xFAFF),   # CJK 兼容区
    (0x20000, 0x2FA1F),  # CJK 扩展 B-F
)


class DownloadError(RuntimeError):
    """下载/校验失败（重试与回退都失败后抛出）。"""


# ----------------------------------------------------------------------
# 基础网络请求
# ----------------------------------------------------------------------
def _urlopen(url: str, timeout: float):
    """发一个 GET 请求并返回 response；测试通过 mock 本函数避免联网。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_bytes(
    url: str,
    retries: int = 3,
    timeout: float = 60.0,
    backoff: float = 1.0,
    expected_sha256: str | None = None,
    verbose: bool = False,
) -> bytes:
    """下载一个 URL 的内容（带重试与可选 sha256 校验）。

    ``retries`` 为总尝试次数（>=1）。校验失败视为本次失败并重试，最终仍失败
    则抛 :class:`DownloadError`。

    注意：Gutenberg 等服务器会间歇性截断响应，``http.client.IncompleteRead`` /
    连接被重置等异常不是 ``OSError`` 的子类，必须显式捕获并重试；读到的字节数
    少于 ``Content-Length`` 时也主动判定失败。
    """
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if verbose:
                print(f"[download] GET {url} (尝试 {attempt}/{attempts})")
            with _urlopen(url, timeout) as response:
                # 必须在读取前记录声明长度：http.client 响应读完后 length 归零。
                expected_length = getattr(response, "length", None)
                if expected_length is None:
                    headers = getattr(response, "headers", None)
                    raw_length = headers.get("Content-Length") if headers is not None else None
                    if raw_length is not None:
                        try:
                            expected_length = int(raw_length)
                        except (TypeError, ValueError):
                            expected_length = None
                # 限长读取：多读 1 字节用于判定是否超出上限。
                data = response.read(MAX_DOWNLOAD_BYTES + 1)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise DownloadError(
                        f"响应超过大小上限 {MAX_DOWNLOAD_BYTES} 字节：{url}"
                    )
                if expected_length is not None and len(data) < int(expected_length):
                    raise DownloadError(
                        f"响应被截断：收到 {len(data)} 字节，Content-Length 声明 {expected_length} 字节：{url}"
                    )
            if not data:
                raise DownloadError(f"下载内容为空：{url}")
            if expected_sha256:
                actual = sha256_bytes(data)
                if actual != expected_sha256:
                    raise DownloadError(
                        f"sha256 校验失败：{url}\n  期望 {expected_sha256}\n  实际 {actual}"
                    )
            return data
        except (
            DownloadError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            http.client.HTTPException,  # IncompleteRead / RemoteDisconnected / BadStatusLine
            OSError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_error = exc
            if verbose:
                print(f"[download] 失败（尝试 {attempt}/{attempts}）：{exc}")
            if attempt < attempts:
                time.sleep(max(0.0, backoff) * attempt)
    raise DownloadError(f"下载失败（已重试 {attempts} 次）：{url}；最后错误：{last_error}")


def _read_cached_text(path: Path) -> str | None:
    """读取缓存文本；不存在/解码失败/内容为空返回 None。"""
    if not path.is_file():
        return None
    try:
        text = decode_text(path.read_bytes())
    except (OSError, ValueError):
        return None
    return text if text.strip() else None


def decode_text(data: bytes) -> str:
    """按 UTF-8(BOM) -> GB18030 的顺序严格解码文本。

    不做 latin-1 兜底：它能把任意字节"解码成功"，坏字节语料会静默变成
    乱码进入词表。无法解码时抛 ValueError，由调用方决定如何处理。
    """
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本无法按 UTF-8 或 GB18030 解码（文件可能损坏或编码不受支持）")


def load_local_text(path: str | Path) -> str:
    """读取本地文本文件并做统一换行处理（``--input`` 兜底路径）。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到本地文本文件：{file_path}")
    text = clean_corpus_text(decode_text(file_path.read_bytes()))
    if not text.strip():
        raise ValueError(f"本地文本为空：{file_path}")
    return text


def clean_corpus_text(text: str) -> str:
    """语料统一清洗：去 BOM/NUL 与控制字符、统一换行为 ``\\n``。"""
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop C0/C1 control characters (keep \n and \t) so they never enter the
    # character vocabulary; DEL and the C1 range are included.
    return "".join(
        ch for ch in text
        if ch in "\n\t" or not (0x00 <= ord(ch) <= 0x1F or 0x7F <= ord(ch) <= 0x9F)
    )


# ----------------------------------------------------------------------
# 英文：Tiny Shakespeare
# ----------------------------------------------------------------------
def download_shakespeare(
    download_dir: str | Path,
    urls: Sequence[str] | None = None,
    expected_sha256: str | None = SHAKESPEARE_SHA256,
    retries: int = 3,
    timeout: float = 60.0,
    verbose: bool = False,
) -> Path:
    """下载/复用 Tiny Shakespeare 原始文本，返回缓存文件路径。

    缓存文件名为 ``<download_dir>/shakespeare_en.txt``；命中且 sha256 正确时
    直接复用（不会联网）。
    """
    dest = Path(download_dir) / "shakespeare_en.txt"
    urls = tuple(urls) if urls else SHAKESPEARE_URLS

    if dest.is_file() and dest.stat().st_size > 0:
        cached_ok = expected_sha256 is None or sha256_file(dest) == expected_sha256
        if cached_ok:
            if verbose:
                print(f"[download] 复用缓存：{dest}")
            return dest
        if verbose:
            print(f"[download] 缓存校验失败，重新下载：{dest}")

    errors: list[str] = []
    for url in urls:
        try:
            data = fetch_bytes(
                url,
                retries=retries,
                timeout=timeout,
                expected_sha256=expected_sha256,
                verbose=verbose,
            )
            atomic_write_bytes(dest, data)
            if expected_sha256 and sha256_file(dest) != expected_sha256:
                raise DownloadError(f"落盘后 sha256 不一致：{dest}")
            if verbose:
                print(f"[download] 已保存：{dest}（{dest.stat().st_size} 字节）")
            return dest
        except DownloadError as exc:
            errors.append(f"{url}: {exc}")
    raise DownloadError("Tiny Shakespeare 所有下载源均失败：\n  " + "\n  ".join(errors))


# ----------------------------------------------------------------------
# 中文：Project Gutenberg 公版书
# ----------------------------------------------------------------------
def strip_gutenberg_markers(raw: str) -> str:
    """去掉 Gutenberg 页眉/页脚，返回正文。

    START/END 标记缺失说明文件不完整（常见于网络截断），抛
    :class:`DownloadError`，由上层重新下载或换镜像。
    """
    text = clean_corpus_text(raw)
    start_match = _START_RE.search(text)
    if start_match is None:
        raise DownloadError("Gutenberg 文本缺少 *** START OF ... *** 标记（可能下载不完整）")
    body = text[start_match.end():]
    end_match = _END_RE.search(body)
    if end_match is not None:
        body = body[: end_match.start()]
    else:
        loose_match = _END_LOOSE_RE.search(body)
        if loose_match is None:
            raise DownloadError("Gutenberg 文本缺少 *** END OF ... *** 标记（可能下载不完整）")
        body = body[: loose_match.start()]
    return body.strip("\n")


def _book_urls(book_id: str, template: str | None = None) -> list[str]:
    """生成一本书的候选 URL 列表（显式模板优先，随后为 Gutenberg 官方回退）。"""
    if template:
        if "{id}" not in template:
            raise DownloadError(
                "中文书 URL 模板必须包含 {id} 占位符，例如 "
                "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
            )
        return [template.format(id=book_id)]
    urls = [GUTENBERG_URL_TEMPLATE.format(id=book_id)]
    urls.extend(item.format(id=book_id) for item in GUTENBERG_FALLBACK_TEMPLATES)
    return urls


def _book_title(raw: str) -> str:
    """在正文前 4000 字符内匹配书名标记，取不到时返回空串。"""
    match = _TITLE_RE.search(raw[:4000])
    return match.group(1) if match else ""


def fetch_zh_book(
    book_id: str,
    download_dir: str | Path,
    template: str | None = None,
    retries: int = 3,
    timeout: float = 120.0,
    min_chars: int = 1000,
    verbose: bool = False,
) -> dict:
    """下载（或复用缓存）单本 Gutenberg 书，返回正文与元数据。

    缓存为原始文件 ``<download_dir>/gutenberg_<id>.txt``；缓存清洗后若缺标记
    或正文过短，会删除缓存并重新下载（防止把截断文件当缓存）。
    """
    cache = Path(download_dir) / f"gutenberg_{book_id}.txt"
    urls = _book_urls(book_id, template)
    raw_text: str | None = None

    cached_raw: str | None = None
    if cache.is_file():
        try:
            cached_raw = cache.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            cached_raw = None  # 损坏缓存按未命中处理，触发重新下载
    if cached_raw is not None and cached_raw.strip():
        try:
            body = strip_gutenberg_markers(cached_raw)
            if len(body) >= min_chars:
                raw_text = cached_raw
                if verbose:
                    print(f"[download] 复用 Gutenberg 缓存 #{book_id}：{cache}")
            else:
                if verbose:
                    print(f"[download] 缓存正文过短（{len(body)} 字符），重新下载 #{book_id}")
        except DownloadError:
            if verbose:
                print(f"[download] 缓存缺标记，重新下载 #{book_id}")

    if raw_text is None:
        errors: list[str] = []
        for url in urls:
            try:
                data = fetch_bytes(url, retries=retries, timeout=timeout, verbose=verbose)
                raw_text = decode_text(data)
                body = strip_gutenberg_markers(raw_text)  # 立刻做完整性校验
                if len(body) < min_chars:
                    raise DownloadError(f"正文过短（{len(body)} < {min_chars} 字符），疑似截断")
                atomic_write_bytes(cache, data)
                break
            except (DownloadError, UnicodeDecodeError) as exc:
                errors.append(f"{url}: {exc}")
                raw_text = None
        if raw_text is None:
            raise DownloadError(f"Gutenberg #{book_id} 所有下载源均失败：\n  " + "\n  ".join(errors))

    body = strip_gutenberg_markers(raw_text)
    raw_bytes = cache.stat().st_size if cache.is_file() else len(raw_text.encode("utf-8"))
    if raw_bytes < len(body.encode("utf-8")):
        raise DownloadError(f"Gutenberg #{book_id} 原始字节数异常：{raw_bytes} 字节")
    return {
        "id": str(book_id),
        "url": urls[0],
        "cache_path": str(cache),
        "raw_sha256": sha256_file(cache) if cache.is_file() else sha256_bytes(raw_text.encode("utf-8")),
        "raw_bytes": raw_bytes,
        "title": _book_title(raw_text),
        "chars": len(body),
        "markers_ok": True,
        "text": body,
    }


def cjk_ratio(text: str) -> float:
    """返回 CJK 字符占非空白字符的比例（用于中文语料完整性抽查）。"""
    total = 0
    cjk = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        code = ord(ch)
        for low, high in _CJK_RANGES:
            if low <= code <= high:
                cjk += 1
                break
    return (cjk / total) if total else 0.0


def download_zh_gutenberg(
    download_dir: str | Path,
    book_ids: Sequence[str] | Iterable[str] | None = None,
    template: str | None = None,
    retries: int = 3,
    timeout: float = 120.0,
    min_chars_per_book: int = 1000,
    min_cjk_ratio: float = 0.01,
    verbose: bool = False,
) -> dict:
    """下载/复用中文 Gutenberg 书单，合并为一份 UTF-8 文本。

    返回字典包含 ``path``（合并后的 ``zh_gutenberg.txt``）、``sha256``、
    ``chars``、``cjk_ratio``、``books``（逐本统计）等，供 ``prepare.py`` 写
    ``meta.json``。
    """
    output_dir = Path(download_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(item).strip() for item in (book_ids or GUTENBERG_BOOK_IDS) if str(item).strip()]
    if not ids:
        raise DownloadError("中文书单为空（检查 zh_books / --zh_books）")

    books: list[dict] = []
    texts: list[str] = []
    for book_id in ids:
        stats = fetch_zh_book(
            book_id,
            output_dir,
            template=template,
            retries=retries,
            timeout=timeout,
            min_chars=min_chars_per_book,
            verbose=verbose,
        )
        texts.append(str(stats.pop("text")))
        books.append(stats)

    combined = "\n\n".join(texts)
    combined = clean_corpus_text(combined).strip("\n") + "\n"
    ratio = cjk_ratio(combined)
    if ratio < min_cjk_ratio:
        raise DownloadError(
            f"合并正文 CJK 比例仅 {ratio:.2%}，低于阈值 {min_cjk_ratio:.2%}；"
            "书单可能不正确或下载内容异常"
        )
    combined_path = output_dir / "zh_gutenberg.txt"
    atomic_write_text(combined_path, combined)
    digest = sha256_file(combined_path)
    if verbose:
        print(
            f"[download] 中文语料已合并：{len(combined):,} 字符，CJK {ratio:.1%}，"
            f"{len(books)} 本书 -> {combined_path}"
        )
    return {
        "path": str(combined_path),
        "sha256": digest,
        "chars": len(combined),
        "cjk_ratio": ratio,
        "book_ids": ids,
        "books": books,
    }


def fetch_zh_book_cached(stats: dict) -> str:
    """从 :func:`fetch_zh_book` 返回的缓存路径重新读出正文（调试/容错用）。

    正常流程不需要它：``download_zh_gutenberg`` 在下载时已拿到正文。
    """
    cache = Path(stats["cache_path"])
    raw = cache.read_text(encoding="utf-8")
    return strip_gutenberg_markers(raw)


# ----------------------------------------------------------------------
# 公开 SFT 数据：alpaca-zh（只缓存不提交）
# ----------------------------------------------------------------------
def download_alpaca_zh(
    download_dir: str | Path,
    url: str = ALPACA_ZH_URL,
    retries: int = 3,
    timeout: float = 180.0,
    verbose: bool = False,
    expected_sha256: str | None = None,
) -> Path:
    """下载/复用 alpaca-zh 原始 JSON（约 35MB），返回缓存路径。

    缓存命中条件：文件存在、可解析为 JSON 数组且非空。官方源下载按
    ``ALPACA_ZH_SHA256`` 钉住完整性（自定义 url 不强制，可显式传
    ``expected_sha256``）；已有缓存不强制哈希（离线优先），但校验不符
    时给出警告。该数据仅用于研究（CC BY-NC 4.0），不得提交仓库。
    """
    if expected_sha256 is None and url == ALPACA_ZH_URL:
        expected_sha256 = ALPACA_ZH_SHA256
    dest = Path(download_dir) / ALPACA_ZH_FILENAME
    if dest.is_file() and dest.stat().st_size > 0:
        try:
            data = read_json(dest)
            if isinstance(data, list) and data:
                if expected_sha256 and sha256_file(dest) != expected_sha256:
                    print(
                        f"[download] 警告：alpaca-zh 缓存与钉住的 sha256 不符：{dest}；"
                        "如需官方原始数据，请删除缓存后重新下载。",
                        file=sys.stderr,
                    )
                if verbose:
                    print(f"[download] 复用 alpaca-zh 缓存：{dest}（{len(data):,} 条）")
                return dest
        except (OSError, ValueError):
            pass
        if verbose:
            print(f"[download] alpaca-zh 缓存损坏，重新下载：{dest}")

    data_bytes = fetch_bytes(
        url, retries=retries, timeout=timeout, verbose=verbose, expected_sha256=expected_sha256
    )
    try:
        parsed = json.loads(data_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"alpaca-zh 下载内容不是合法 JSON：{exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise DownloadError("alpaca-zh 下载内容不是非空 JSON 数组")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(dest, data_bytes)
    if verbose:
        print(f"[download] 已缓存 alpaca-zh：{dest}（{len(parsed):,} 条，{dest.stat().st_size:,} 字节）")
    return dest


__all__ = [
    "USER_AGENT",
    "SHAKESPEARE_SHA256",
    "SHAKESPEARE_SIZE",
    "SHAKESPEARE_URLS",
    "GUTENBERG_BOOK_IDS",
    "GUTENBERG_URL_TEMPLATE",
    "ALPACA_ZH_URL",
    "ALPACA_ZH_FILENAME",
    "ALPACA_ZH_LICENSE",
    "DownloadError",
    "fetch_bytes",
    "decode_text",
    "load_local_text",
    "clean_corpus_text",
    "download_shakespeare",
    "strip_gutenberg_markers",
    "fetch_zh_book",
    "fetch_zh_book_cached",
    "cjk_ratio",
    "download_zh_gutenberg",
    "download_alpaca_zh",
]
