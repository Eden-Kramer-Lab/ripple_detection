"""Fetch the inputs of the reference-recording checks into a cache, with a manifest.

Everything lands under the cache directory, ``$RIPPLE_REFERENCE_CACHE`` or
``~/.cache/ripple_detection/reference_recordings``, never in the repository.
Each fetch appends a record to ``<cache>/manifest.json`` (a list, rewritten
atomically): kind, url, local path (relative to the cache), bytes, SHA-256, MD5
where computed, the checksum it was checked against and where that came from,
the retrieval time (UTC) and the versions in use. Every download is written to
``<name>.part``, verified, then renamed; a mismatch removes it and raises.

CRCNS credentials are read only from ``CRCNS_USERNAME`` and ``CRCNS_PASSWORD``;
they are never printed, logged or written, and the session cookie stays in memory.

Command line (``python examples/reference_recordings/fetch.py <command> ...``)::

    url      URL DEST [--sha256 H] [--md5 H] [--size N]
    wayback  ORIGINAL_URL TIMESTAMP DEST [--sha256 H] [--head-bytes N]
    dandi    DANDISET VERSION PATH_OR_ASSET_ID
    crcns    DATASET PATH_OR_GLOB ... [--max-mb N]

``DEST`` is relative to the cache. Opening an NWB file remotely lives in
``nwb.py``, which needs ``remfile`` and ``h5py``; this module needs neither.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import http.cookiejar
import importlib.metadata
import json
import os
import platform
import re
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

CACHE_ENV = "RIPPLE_REFERENCE_CACHE"
DEFAULT_CACHE = Path("~/.cache/ripple_detection/reference_recordings")
USER_AGENT = "ripple_detection reference-recordings fetch"
CHUNK = 1 << 20

WAYBACK_BASE = "https://web.archive.org/web"
DANDI_API = "https://api.dandiarchive.org/api"
CRCNS_SITE = "https://crcns.org"
CRCNS_LOGIN_URL = f"{CRCNS_SITE}/login_form"
CRCNS_DOWNLOAD_BASE = "https://download.crcns.org"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def cache_dir() -> Path:
    """Return the cache directory (``$RIPPLE_REFERENCE_CACHE`` or the default).

    Returns
    -------
    pathlib.Path
        The expanded directory; it is not created.
    """
    return Path(os.environ.get(CACHE_ENV) or DEFAULT_CACHE).expanduser()


def _versions(*packages: str) -> dict[str, str]:
    """Python's version and those of the named packages that are installed."""
    found = {"python": platform.python_version()}
    for name in packages:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            found[name] = importlib.metadata.version(name)
    return found


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digests(path: Path) -> tuple[str, str]:
    """SHA-256 and MD5 hex digests of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash, read in 1 MiB chunks.

    Returns
    -------
    sha256, md5 : str
        Lower-case hex digests.
    """
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def append_manifest(record: dict[str, Any], cache: Path | None = None) -> None:
    """Append a record to ``<cache>/manifest.json``, rewriting the file atomically.

    Parameters
    ----------
    record : dict
        JSON-serialisable fetch record.
    cache : pathlib.Path, optional
        Cache directory; default `cache_dir()`.
    """
    cache = cache_dir() if cache is None else cache
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    if not isinstance(records, list):
        msg = f"{manifest} does not hold a list of records"
        raise ValueError(msg)
    records.append(record)
    partial = manifest.with_name(manifest.name + ".part")
    partial.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    partial.replace(manifest)


def read_manifest(cache: Path | None = None) -> list[dict[str, Any]]:
    """Read ``<cache>/manifest.json`` (an empty list when there is none)."""
    manifest = (cache_dir() if cache is None else cache) / "manifest.json"
    if not manifest.exists():
        return []
    records: list[dict[str, Any]] = json.loads(manifest.read_text(encoding="utf-8"))
    return records


def _destination(dest: str | Path, cache: Path) -> Path:
    """Resolve ``dest`` inside the cache, refusing paths that leave it."""
    relative = Path(dest)
    if relative.is_absolute() or ".." in relative.parts:
        msg = f"destination {str(dest)!r} must be a relative path inside the cache"
        raise ValueError(msg)
    return cache / relative


def _stream_to_part(response: Any, partial: Path, max_bytes: int | None = None) -> int:
    """Write a response body to ``partial``; return the bytes written."""
    partial.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with partial.open("wb") as f:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                n += len(chunk)
                if max_bytes is not None and n > max_bytes:
                    msg = f"passed the cap of {max_bytes} bytes"
                    raise ValueError(msg)
                f.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return n


def _verify(
    partial: Path,
    expected_size: int | None,
    expected_sha256: str | None,
    expected_md5: str | None,
    label: str,
) -> tuple[int, str, str]:
    """Check ``partial`` against the expected values; delete it and raise on a mismatch."""
    size = partial.stat().st_size
    sha, md5 = digests(partial)
    problems = []
    if expected_size is not None and size != expected_size:
        problems.append(f"size {size} != expected {expected_size}")
    if expected_sha256 is not None and sha != expected_sha256.lower():
        problems.append(f"sha256 {sha} != expected {expected_sha256}")
    if expected_md5 is not None and md5 != expected_md5.lower():
        problems.append(f"md5 {md5} != expected {expected_md5}")
    if problems:
        partial.unlink()
        raise ValueError(f"{label}: " + "; ".join(problems) + "; the download was removed")
    return size, sha, md5


def _expected_record(
    size: int | None, sha256: str | None, md5: str | None, source: str | None
) -> dict[str, Any]:
    return {"bytes": size, "sha256": sha256, "md5": md5, "source": source}


def fetch_url(
    url: str,
    dest: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
    expected_source: str | None = None,
    cache: Path | None = None,
) -> dict[str, Any]:
    """Download one file over HTTPS (or ``file://``), verify it and record it.

    Parameters
    ----------
    url : str
        Source URL.
    dest : str or pathlib.Path
        Path relative to the cache.
    expected_size, expected_sha256, expected_md5 : optional
        Checks; any given must match.
    expected_source : str, optional
        Where the expected values came from, kept in the manifest.
    cache : pathlib.Path, optional
        Cache directory; default `cache_dir()`.

    Returns
    -------
    dict
        The manifest record; ``path`` is relative to the cache.

    Raises
    ------
    ValueError
        On a size or checksum mismatch; the file is removed and no record is written.
    """
    cache = cache_dir() if cache is None else cache
    target = _destination(dest, cache)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        _stream_to_part(response, partial)
    size, sha, md5 = _verify(partial, expected_size, expected_sha256, expected_md5, url)
    partial.replace(target)
    record = {
        "kind": "url",
        "url": url,
        "path": str(Path(dest)),
        "bytes": size,
        "sha256": sha,
        "md5": md5,
        "expected": _expected_record(
            expected_size, expected_sha256, expected_md5, expected_source
        ),
        "retrieved": _now(),
        "versions": _versions(),
    }
    append_manifest(record, cache)
    return record


_WAYBACK_LENGTH_HEADERS = (
    "x-archive-orig-x-crawler-content-length",
    "x-archive-orig-content-length",
)


def _wayback_declared_length(headers: Mapping[str, str]) -> tuple[int, str] | None:
    """The original length and the header that gave it, or None when neither is present."""
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for name in _WAYBACK_LENGTH_HEADERS:
        raw = lowered.get(name)
        if raw is not None:
            try:
                return int(str(raw).strip()), name
            except ValueError as error:
                msg = f"{name} is not an integer: {raw!r}"
                raise ValueError(msg) from error
    return None


def wayback_capture_timestamp(url: str) -> str | None:
    """The 14-digit capture timestamp in a ``web.archive.org/web/<timestamp>[id_]/...`` URL.

    Wayback redirects a request to the nearest capture, so the URL a response was
    served from (``response.geturl()``) names the capture actually returned.

    Parameters
    ----------
    url : str
        A Wayback URL.

    Returns
    -------
    str or None
        The timestamp, or None when the URL has none.
    """
    match = re.search(r"/web/(\d{14})(?:[a-z]{2}_)?/", url)
    return match.group(1) if match else None


def wayback_original_length(headers: Mapping[str, str], n_received: int) -> tuple[int, str]:
    """Apply the Wayback length rule and return the original length and its header.

    The archive keeps only the first 1 MiB of large files, so a capture is usable
    only if the original length is declared and equals the bytes received. For a
    truncated capture ``x-archive-orig-content-length`` can describe the stored
    record, while ``x-archive-orig-x-crawler-content-length`` carries the
    crawler's true length, so the latter is taken when present.

    Parameters
    ----------
    headers : mapping of str to str
        Response headers (names compared case-insensitively).
    n_received : int
        Bytes in the response body.

    Returns
    -------
    original_length : int
        The original length, equal to ``n_received``.
    header : str
        Lower-case name of the header that supplied it.

    Raises
    ------
    ValueError
        If neither header is present or one is malformed, or the body is shorter or
        longer than the declared length.
    """
    declared = _wayback_declared_length(headers)
    if declared is None:
        msg = (
            "capture has neither x-archive-orig-x-crawler-content-length nor "
            "x-archive-orig-content-length; its completeness cannot be checked"
        )
        raise ValueError(msg)
    original, header = declared
    if n_received < original:
        msg = f"truncated capture: received {n_received} bytes of the original {original} ({header})"
        raise ValueError(msg)
    if n_received > original:
        msg = f"capture is longer than the original: {n_received} bytes against {original} ({header})"
        raise ValueError(msg)
    return original, header


def fetch_wayback(
    original_url: str,
    timestamp: str,
    dest: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_source: str | None = None,
    head_bytes: int | None = None,
    allow_other_capture: bool = False,
    cache: Path | None = None,
) -> dict[str, Any]:
    """Download an Internet Archive capture's unmodified bytes, refusing truncated ones.

    Requests ``https://web.archive.org/web/<timestamp>id_/<original_url>``. The file
    is kept only if `wayback_original_length` accepts it.

    With ``head_bytes`` the request carries ``Range: bytes=0-(head_bytes-1)``, the
    first bytes are stored as ``<dest>.head`` and the length rule is skipped. Use it
    only to compare the first samples of a raw binary with another copy; such a file
    is never parsed as a whole.

    Parameters
    ----------
    original_url : str
        The URL that was archived.
    timestamp : str
        Capture timestamp, ``YYYYMMDDhhmmss``.
    dest : str or pathlib.Path
        Path relative to the cache.
    expected_sha256 : str, optional
        Checked after the length rule (ignored in head mode).
    expected_source : str, optional
        Where ``expected_sha256`` came from.
    head_bytes : int, optional
        Fetch only this many leading bytes (head mode); a short read raises (the
        original length, when smaller, is the expected size).
    allow_other_capture : bool
        Wayback redirects to the nearest capture. By default a response served from a
        different capture than ``timestamp`` raises; with True it is kept and both
        timestamps are recorded.
    cache : pathlib.Path, optional
        Cache directory; default `cache_dir()`.

    Returns
    -------
    dict
        The manifest record, with the requested and served capture URL and timestamp, original length and
        the header it came from, and the original ``Last-Modified``; head mode adds
        ``head=true`` and ``range``.
    """
    cache = cache_dir() if cache is None else cache
    if head_bytes is not None and head_bytes < 1:
        msg = "head_bytes must be positive"
        raise ValueError(msg)
    target = _destination(dest, cache)
    if head_bytes is not None:
        target = target.with_name(target.name + ".head")
    partial = target.with_name(target.name + ".part")
    capture_url = f"{WAYBACK_BASE}/{timestamp}id_/{original_url}"
    request_headers = {"User-Agent": USER_AGENT}
    if head_bytes is not None:
        request_headers["Range"] = f"bytes=0-{head_bytes - 1}"
    request = urllib.request.Request(capture_url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        served_url = response.geturl()
        if head_bytes is None:
            n = _stream_to_part(response, partial)
        else:
            partial.parent.mkdir(parents=True, exist_ok=True)
            chunks, remaining = [], head_bytes
            while remaining:
                chunk = response.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            partial.write_bytes(b"".join(chunks))
            n = partial.stat().st_size
        headers = dict(response.headers.items())
    served_timestamp = wayback_capture_timestamp(served_url)
    if served_timestamp != timestamp and not allow_other_capture:
        partial.unlink(missing_ok=True)
        msg = (
            f"{capture_url}: served from capture {served_timestamp} ({served_url}), "
            f"not the requested {timestamp}; the download was removed"
        )
        raise ValueError(msg)
    lowered = {k.lower(): v for k, v in headers.items()}
    record: dict[str, Any] = {
        "kind": "wayback",
        "url": capture_url,
        "served_url": served_url,
        "original_url": original_url,
        "requested_timestamp": timestamp,
        "wayback_timestamp": served_timestamp,
        "original_last_modified": lowered.get("x-archive-orig-last-modified"),
    }
    if head_bytes is None:
        try:
            original_length, length_header = wayback_original_length(headers, n)
        except ValueError as error:
            partial.unlink(missing_ok=True)
            msg = f"{capture_url}: {error}; the capture was removed"
            raise ValueError(msg) from error
        size, sha, md5 = _verify(partial, original_length, expected_sha256, None, capture_url)
        expected = _expected_record(original_length, expected_sha256, None, expected_source)
        record["head"] = False
    else:
        declared = _wayback_declared_length(headers)
        original_length, length_header = declared or (None, None)
        wanted = head_bytes if original_length is None else min(head_bytes, original_length)
        if n != wanted:
            partial.unlink(missing_ok=True)
            msg = f"{capture_url}: head read {n} bytes, expected {wanted}; the download was removed"
            raise ValueError(msg)
        size = n
        sha, md5 = digests(partial)
        expected = _expected_record(None, None, None, None)
        record["head"] = True
        record["range"] = f"bytes=0-{head_bytes - 1}"
    partial.replace(target)
    record.update(
        original_content_length=original_length,
        original_length_header=length_header,
        path=str(target.relative_to(cache)),
        bytes=size,
        sha256=sha,
        md5=md5,
        expected=expected,
        retrieved=_now(),
        versions=_versions(),
    )
    append_manifest(record, cache)
    return record


def _get_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def dandi_list_assets(dandiset: str, version: str, path_prefix: str) -> list[dict[str, Any]]:
    """List a dandiset version's assets whose path starts with a prefix.

    Parameters
    ----------
    dandiset : str
        Six-digit dandiset identifier, for example ``"000059"``.
    version : str
        Version, ``"draft"`` or a released one.
    path_prefix : str
        Asset path prefix, for example a subject folder.

    Returns
    -------
    list of dict
        The API's asset summaries (``asset_id``, ``path``, ``size``, ...), all pages.
    """
    query = urllib.parse.urlencode({"path": path_prefix, "page_size": 200})
    url: str | None = f"{DANDI_API}/dandisets/{dandiset}/versions/{version}/assets/?{query}"
    found: list[dict[str, Any]] = []
    while url:
        page = _get_json(url)
        found.extend(page["results"])
        url = page.get("next")
    return found


def dandi_asset(
    dandiset: str,
    version: str,
    path_or_asset_id: str,
    *,
    cache: Path | None = None,
) -> dict[str, Any]:
    """Look up a DANDI asset and record it (kind ``dandi-stream``).

    Streamed slices of the file are not hashed; the record says so and carries the
    archive's own ``dandi:sha2-256``.

    Parameters
    ----------
    dandiset : str
        Six-digit dandiset identifier.
    version : str
        ``"draft"`` or a released version.
    path_or_asset_id : str
        An asset UUID, or the asset's exact path in the dandiset.
    cache : pathlib.Path, optional
        Cache directory; default `cache_dir()`.

    Returns
    -------
    dict
        The manifest record: ``asset_id``, ``asset_path``, ``bytes`` (the asset's
        size), ``sha256`` (the archive's digest), ``content_url`` (S3).

    Raises
    ------
    ValueError
        If no asset has exactly that path.
    """
    cache = cache_dir() if cache is None else cache
    if _UUID.fullmatch(path_or_asset_id):
        asset_id = path_or_asset_id
    else:
        hits = [
            a
            for a in dandi_list_assets(dandiset, version, path_or_asset_id)
            if a["path"] == path_or_asset_id
        ]
        if len(hits) != 1:
            msg = f"dandiset {dandiset} {version}: {len(hits)} assets with path {path_or_asset_id!r}"
            raise ValueError(msg)
        asset_id = hits[0]["asset_id"]
    base = f"{DANDI_API}/dandisets/{dandiset}/versions/{version}/assets/{asset_id}"
    info = _get_json(f"{base}/")
    content_urls = [u for u in info.get("contentUrl", []) if "s3" in u]
    if not content_urls:
        msg = f"asset {asset_id} lists no S3 content URL"
        raise ValueError(msg)
    sha = info.get("digest", {}).get("dandi:sha2-256")
    record = {
        "kind": "dandi-stream",
        "url": f"{DANDI_API}/assets/{asset_id}/download/",
        "dandiset": dandiset,
        "dandiset_version": version,
        "asset_id": asset_id,
        "asset_path": info["path"],
        "content_url": content_urls[0],
        "path": None,
        "bytes": info["contentSize"],
        "sha256": sha,
        "md5": None,
        "sha256_note": "the archive's dandi:sha2-256; streamed slices are not hashed",
        "expected": _expected_record(None, None, None, None),
        "retrieved": _now(),
        "versions": _versions("remfile", "h5py"),
    }
    append_manifest(record, cache)
    return record


# --- CRCNS: pure helpers --------------------------------------------------------------


def parse_crcns_filelist(text: str) -> dict[str, int]:
    """Map each path in a CRCNS ``filelist.txt`` to its size in bytes.

    Lines look like `` code.zip\\t108361 (105.8 KB)``; ``#`` lines that are
    comments (a ``#`` followed by prose) do not match a path and a size, so they
    are skipped, while a commented-out file (``#path<TAB>size``) is read like the
    others.

    Parameters
    ----------
    text : str
        The file's contents.

    Returns
    -------
    dict of str to int
        Path to size in bytes.
    """
    sizes: dict[str, int] = {}
    pattern = re.compile(r"^[ +#]\s*(\S+)\s+(\d+)(?:\s*\([\d.]+ \w+\))?\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            sizes[match.group(1)] = int(match.group(2))
    return sizes


def parse_crcns_checksums(text: str) -> dict[str, str]:
    """Map each path in a CRCNS ``checksums.md5`` to its MD5 hex digest.

    Parameters
    ----------
    text : str
        The file's contents, ``<md5>  <path>`` per line (a leading ``*`` or ``./``
        on the path is dropped).

    Returns
    -------
    dict of str to str
        Path to lower-case MD5.
    """
    sums: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{32}", parts[0]):
            sums[parts[1].strip().lstrip("*").removeprefix("./")] = parts[0].lower()
    return sums


def is_crcns_login_page(content_type: str, first_bytes: bytes) -> bool:
    """Whether a download response is CRCNS's login page rather than the file.

    Parameters
    ----------
    content_type : str
        The response's ``Content-Type``.
    first_bytes : bytes
        The start of the body.
    """
    return content_type.startswith("text/html") and b"Login below" in first_bytes


def select_paths(sizes: Mapping[str, int], patterns: Iterable[str]) -> list[str]:
    """Paths in a file list matching exact names or glob patterns, each once, in order.

    Raises
    ------
    ValueError
        If a pattern matches nothing.
    """
    chosen: list[str] = []
    for pattern in patterns:
        pattern = pattern.lstrip("/")
        hits = [p for p in sizes if p == pattern or fnmatch.fnmatchcase(p, pattern)]
        if not hits:
            msg = f"{pattern!r} matches nothing in the file list"
            raise ValueError(msg)
        chosen.extend(h for h in hits if h not in chosen)
    return chosen


class CrcnsSession:
    """An in-memory cookie session against crcns.org and download.crcns.org.

    Credentials come from ``CRCNS_USERNAME`` and ``CRCNS_PASSWORD`` when `login` is
    called; the cookie jar is never written anywhere.
    """

    def __init__(self) -> None:
        jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self.opener.addheaders = [("User-Agent", USER_AGENT)]

    def text(self, url: str, form: dict[str, str] | None = None) -> str:
        """Fetch a page (POST when ``form`` is given) as text."""
        data = urllib.parse.urlencode(form).encode() if form else None
        with self.opener.open(url, data=data, timeout=60) as response:
            return str(response.read().decode("utf-8", errors="replace"))

    def logged_in(self, dataset: str) -> bool:
        """Whether the dataset page says the session is logged in."""
        plain = re.sub(r"<[^>]*>", " ", self.text(f"{CRCNS_DOWNLOAD_BASE}/{dataset}"))
        if re.search(r"Invalid data set identifier", plain, re.IGNORECASE):
            msg = f"CRCNS says {dataset!r} is not a data set identifier"
            raise ValueError(msg)
        return re.search(r"Logged in as\s+\S+", plain, re.IGNORECASE) is not None

    def login(self, dataset: str) -> None:
        """Log in with the environment's credentials (the crcns.org form, then the download site's)."""
        username = os.environ.get("CRCNS_USERNAME")
        password = os.environ.get("CRCNS_PASSWORD")
        if not username or not password:
            msg = "set CRCNS_USERNAME and CRCNS_PASSWORD in the environment"
            raise RuntimeError(msg)
        self.text(
            CRCNS_LOGIN_URL,
            {
                "came_from": "",
                "form.submitted": "1",
                "js_enabled": "0",
                "cookies_enabled": "",
                "login_name": "",
                "pwd_empty": "0",
                "__ac_name": username,
                "__ac_password": password,
                "submit": "Log in",
            },
        )
        if self.logged_in(dataset):
            return
        self.text(
            f"{CRCNS_DOWNLOAD_BASE}/{dataset}",
            {"fn": dataset, "username": username, "password": password, "submit": "Login"},
        )
        if not self.logged_in(dataset):
            msg = "CRCNS login was not accepted by either form; check the account"
            raise RuntimeError(msg)

    def download(
        self,
        url: str,
        out: Path,
        max_bytes: int | None,
        expected_size: int | None = None,
        expected_md5: str | None = None,
    ) -> tuple[int, str, str]:
        """Download ``url`` (with ``?agent=1``) to ``out.part``, verify it, then rename.

        Returns
        -------
        size, sha256, md5 : int, str, str
            Of the verified file.

        Raises
        ------
        RuntimeError
            If the server returns the login page.
        ValueError
            If the file passes ``max_bytes``, or its size or MD5 differs from the
            expected one (the ``.part`` file is removed).
        """
        partial = out.with_name(out.name + ".part")
        request = urllib.request.Request(f"{url}?agent=1")
        with self.opener.open(request, timeout=120) as response:
            length = response.headers.get("Content-Length")
            if length is not None and max_bytes is not None and int(length) > max_bytes:
                msg = f"{url}: {int(length)} bytes exceeds the cap"
                raise ValueError(msg)
            first = response.read(CHUNK)
            if is_crcns_login_page(response.headers.get("Content-Type", ""), first):
                msg = f"{url}: the server returned the login page"
                raise RuntimeError(msg)
            partial.parent.mkdir(parents=True, exist_ok=True)
            n = len(first)
            try:
                with partial.open("wb") as f:
                    f.write(first)
                    for chunk in iter(lambda: response.read(CHUNK), b""):
                        n += len(chunk)
                        if max_bytes is not None and n > max_bytes:
                            msg = f"{url}: passed the cap of {max_bytes} bytes"
                            raise ValueError(msg)
                        f.write(chunk)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        result = _verify(partial, expected_size, None, expected_md5, url)
        partial.replace(out)
        return result


def crcns_fetch(
    dataset: str,
    paths: Iterable[str],
    max_mb: float = 200.0,
    *,
    cache: Path | None = None,
    session: CrcnsSession | None = None,
) -> list[dict[str, Any]]:
    """Download files of a CRCNS dataset, checking each against ``checksums.md5``.

    Files go to ``<cache>/crcns/<dataset>/<path>``, with the dataset's
    ``filelist.txt`` and ``checksums.md5`` beside them. A file already present at
    its listed size is verified and recorded again rather than downloaded.

    Parameters
    ----------
    dataset : str
        CRCNS identifier, for example ``"hc-14"``.
    paths : iterable of str
        Paths or glob patterns in the file list.
    max_mb : float
        Per-file cap in MB (1e6 bytes); a larger file is refused.
    cache : pathlib.Path, optional
        Cache directory; default `cache_dir()`.
    session : CrcnsSession, optional
        A logged-in session; one is created and logged in from the environment if omitted.

    Returns
    -------
    list of dict
        One manifest record per file.

    Raises
    ------
    ValueError
        On a pattern matching nothing, a file over the cap, or an MD5 mismatch (the
        file is removed).
    """
    cache = cache_dir() if cache is None else cache
    if session is None:
        session = CrcnsSession()
        session.login(dataset)
    folder = cache / "crcns" / dataset
    for name in ("filelist.txt", "checksums.md5"):
        if not (folder / name).exists():
            session.download(f"{CRCNS_DOWNLOAD_BASE}/{dataset}/{name}", folder / name, None)
    sizes = parse_crcns_filelist((folder / "filelist.txt").read_text(encoding="utf-8"))
    sums = parse_crcns_checksums((folder / "checksums.md5").read_text(encoding="utf-8"))
    cap = int(max_mb * 1e6)
    records = []
    for path in select_paths(sizes, paths):
        if sizes[path] > cap:
            msg = f"{path}: {sizes[path]} bytes is over the {cap}-byte cap"
            raise ValueError(msg)
        out = folder / path
        out.parent.mkdir(parents=True, exist_ok=True)
        listed = sums.get(path)
        if out.exists() and out.stat().st_size == sizes[path]:
            size, sha, md5 = digests_and_size(out)
            if listed is not None and md5 != listed:
                out.unlink()
                msg = f"{path}: cached file has md5 {md5}, listed {listed}; removed"
                raise ValueError(msg)
        else:
            size, sha, md5 = session.download(
                f"{CRCNS_DOWNLOAD_BASE}/{dataset}/{path}", out, cap, sizes[path], listed
            )
        record = {
            "kind": "crcns",
            "url": f"{CRCNS_DOWNLOAD_BASE}/{dataset}/{path}",
            "path": str(out.relative_to(cache)),
            "bytes": size,
            "sha256": sha,
            "md5": md5,
            "expected": _expected_record(
                sizes[path], None, listed, f"{CRCNS_DOWNLOAD_BASE}/{dataset}/checksums.md5"
            ),
            "retrieved": _now(),
            "versions": _versions(),
        }
        append_manifest(record, cache)
        records.append(record)
    return records


def digests_and_size(path: Path) -> tuple[int, str, str]:
    """Size, SHA-256 and MD5 of a file."""
    sha, md5 = digests(path)
    return path.stat().st_size, sha, md5


def extract_members(
    tar_path: str | Path,
    patterns: Iterable[str],
    dest: str | Path,
    *,
    cache: Path | None = None,
) -> list[Path]:
    """Extract only the matching members of a ``.tar`` or ``.tar.gz``.

    Members are read and written one regular file at a time, each through a
    ``.part`` file; names that are absolute or contain ``..`` are refused before
    anything is written. Each extracted file's SHA-256 goes in the manifest with
    the archive as its source.

    Parameters
    ----------
    tar_path : str or pathlib.Path
        The archive.
    patterns : iterable of str
        ``fnmatch`` patterns on the member names.
    dest : str or pathlib.Path
        Directory to extract into.
    cache : pathlib.Path, optional
        Cache directory for the manifest; default `cache_dir()`.

    Returns
    -------
    list of pathlib.Path
        The extracted files, in archive order.

    Raises
    ------
    ValueError
        If a member name is absolute or contains ``..``, or no member matches.
    """
    cache = cache_dir() if cache is None else cache
    tar_path, dest = Path(tar_path), Path(dest)
    patterns = list(patterns)
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:*") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.name.startswith(("/", "\\")):
                msg = f"{tar_path}: member {member.name!r} leaves the destination"
                raise ValueError(msg)
        chosen = [m for m in members if any(fnmatch.fnmatchcase(m.name, p) for p in patterns)]
        if not chosen:
            msg = f"{tar_path}: no member matches {patterns!r}"
            raise ValueError(msg)
        for member in chosen:
            out = dest / member.name
            out.parent.mkdir(parents=True, exist_ok=True)
            partial = out.with_name(out.name + ".part")
            source = archive.extractfile(member)
            if source is None:
                msg = f"{tar_path}: member {member.name!r} cannot be read"
                raise ValueError(msg)
            with source:
                _stream_to_part(source, partial)
            partial.replace(out)
            size, sha, md5 = digests_and_size(out)
            try:
                relative = str(out.relative_to(cache))
            except ValueError:
                relative = str(out)
            append_manifest(
                {
                    "kind": "tar-member",
                    "url": None,
                    "archive": str(tar_path),
                    "member": member.name,
                    "path": relative,
                    "bytes": size,
                    "sha256": sha,
                    "md5": md5,
                    "expected": _expected_record(
                        member.size, None, None, f"member of {tar_path}"
                    ),
                    "retrieved": _now(),
                    "versions": _versions(),
                },
                cache,
            )
            extracted.append(out)
    return extracted


# --- command line ---------------------------------------------------------------------


def _print_record(record: dict[str, Any]) -> None:
    print(f"path   {record.get('path') or record.get('content_url')}")
    print(f"bytes  {record['bytes']}")
    print(f"sha256 {record['sha256']}")


def main(argv: list[str] | None = None) -> None:
    """Run the command line (see the module docstring)."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=None, help="cache directory")
    sub = parser.add_subparsers(dest="command", required=True)
    p_url = sub.add_parser("url", help="download an HTTPS file")
    p_url.add_argument("url")
    p_url.add_argument("dest")
    p_url.add_argument("--sha256")
    p_url.add_argument("--md5")
    p_url.add_argument("--size", type=int)
    p_way = sub.add_parser("wayback", help="download an Internet Archive capture")
    p_way.add_argument("original_url")
    p_way.add_argument("timestamp")
    p_way.add_argument("dest")
    p_way.add_argument("--sha256")
    p_way.add_argument(
        "--head-bytes", type=int, help="fetch only the first N bytes (.head file)"
    )
    p_dandi = sub.add_parser("dandi", help="record a DANDI asset's metadata")
    p_dandi.add_argument("dandiset")
    p_dandi.add_argument("version")
    p_dandi.add_argument("path_or_asset_id")
    p_crcns = sub.add_parser(
        "crcns", help="download CRCNS files (credentials from the environment)"
    )
    p_crcns.add_argument("dataset")
    p_crcns.add_argument("paths", nargs="+")
    p_crcns.add_argument("--max-mb", type=float, default=200.0)
    args = parser.parse_args(argv)

    records: list[dict[str, Any]]
    if args.command == "url":
        records = [
            fetch_url(
                args.url,
                args.dest,
                expected_size=args.size,
                expected_sha256=args.sha256,
                expected_md5=args.md5,
                cache=args.cache,
            )
        ]
    elif args.command == "wayback":
        records = [
            fetch_wayback(
                args.original_url,
                args.timestamp,
                args.dest,
                expected_sha256=args.sha256,
                head_bytes=args.head_bytes,
                cache=args.cache,
            )
        ]
    elif args.command == "dandi":
        records = [
            dandi_asset(args.dandiset, args.version, args.path_or_asset_id, cache=args.cache)
        ]
    else:
        records = crcns_fetch(args.dataset, args.paths, args.max_mb, cache=args.cache)
    for record in records:
        _print_record(record)


if __name__ == "__main__":
    main()
