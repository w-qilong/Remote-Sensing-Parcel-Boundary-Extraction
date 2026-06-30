"""Download and unpack the AI4Boundaries dataset.

The dataset is published as an Apache-style directory listing at:
https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/

By default this script mirrors the remote files under ``datasets/AI4Boundaries``
and extracts supported archives in place.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - useful when running outside this env.
    tqdm = None


BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/"
DEFAULT_OUTPUT_DIR = Path("datasets") / "AI4Boundaries"
ARCHIVE_DIR = "_archives"
REQUEST_TIMEOUT = (10, 120)
CHUNK_SIZE = 1024 * 1024
ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".gz",
)


def _progress(*args, **kwargs):
    if tqdm is None:
        return None
    return tqdm(*args, **kwargs)


def _same_dataset_url(url: str, base_url: str) -> bool:
    """Return True only for URLs inside the AI4Boundaries directory."""

    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return (
        parsed_url.scheme in {"http", "https"}
        and parsed_url.netloc == parsed_base.netloc
        and parsed_url.path.startswith(parsed_base.path)
    )


def _remote_relative_path(url: str, base_url: str) -> Path:
    """Map a remote file URL to a safe relative path below the output root."""

    remote_path = unquote(urlparse(url).path)
    base_path = unquote(urlparse(base_url).path)
    relative = remote_path.removeprefix(base_path).lstrip("/")
    if not relative or relative.startswith("../") or "/../" in f"/{relative}":
        raise ValueError(f"Unsafe remote path: {url}")
    return Path(relative)


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _archive_output_dir(archive_path: Path, archive_root: Path, output_root: Path) -> Path:
    """Extract archive contents beside their mirrored parent directory."""

    relative_parent = archive_path.parent.relative_to(archive_root)
    return output_root / relative_parent


def _safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target_path = (output_dir / member.filename).resolve()
            if not target_path.is_relative_to(output_dir):
                raise ValueError(f"Archive member escapes output directory: {member.filename}")
        archive.extractall(output_dir)


def _safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target_path = (output_dir / member.name).resolve()
            if not target_path.is_relative_to(output_dir):
                raise ValueError(f"Archive member escapes output directory: {member.name}")
        archive.extractall(output_dir)


def _extract_gzip_file(archive_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / archive_path.name.removesuffix(".gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive_path, "rb") as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return output_path


def extract_archive(archive_path: Path, output_dir: Path, *, force: bool = False) -> None:
    """Extract a supported archive and leave a marker for idempotent reruns."""

    marker_path = output_dir / f".{archive_path.name}.extracted"
    if marker_path.exists() and not force:
        print(f"Already extracted: {archive_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    print(f"Extracting {archive_path} -> {output_dir}")
    if name.endswith(".zip"):
        _safe_extract_zip(archive_path, output_dir)
    elif tarfile.is_tarfile(archive_path):
        _safe_extract_tar(archive_path, output_dir)
    elif name.endswith(".gz"):
        _extract_gzip_file(archive_path, output_dir)
    else:
        print(f"Skipping unsupported archive type: {archive_path}")
        return

    marker_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")


def collect_file_urls(base_url: str = BASE_URL, session: requests.Session | None = None) -> list[str]:
    """Recursively collect downloadable files from the dataset index."""

    session = session or requests.Session()
    base_url = base_url.rstrip("/") + "/"
    visited_dirs: set[str] = set()
    file_urls: list[str] = []

    def scrape(directory_url: str) -> None:
        directory_url = directory_url.rstrip("/") + "/"
        if directory_url in visited_dirs:
            return
        visited_dirs.add(directory_url)

        response = session.get(directory_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if (
                not href
                or href.startswith("#")
                or href.startswith("?")
                or href in {"../", "/"}
            ):
                continue

            next_url = urljoin(directory_url, href)
            next_url = next_url.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
            if not _same_dataset_url(next_url, base_url):
                continue

            if next_url.rstrip("/").endswith("AI4BOUNDARIES"):
                continue
            if href.endswith("/"):
                scrape(next_url)
            else:
                file_urls.append(next_url)

    scrape(base_url)
    return sorted(set(file_urls))


def _remote_size(url: str, session: requests.Session) -> int | None:
    try:
        response = session.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return None
    content_length = response.headers.get("Content-Length")
    return int(content_length) if content_length and content_length.isdigit() else None


def download_file(
    url: str,
    output_path: Path,
    session: requests.Session,
    *,
    force: bool = False,
    retries: int = 3,
) -> Path:
    """Stream a URL to disk with retry and simple skip-if-complete behavior."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    remote_size = _remote_size(url, session) if output_path.exists() or tmp_path.exists() else None
    if (
        output_path.exists()
        and not force
        and remote_size is not None
        and output_path.stat().st_size == remote_size
    ):
        print(f"Already downloaded: {output_path}")
        return output_path

    if force and tmp_path.exists():
        tmp_path.unlink()

    last_error: requests.RequestException | None = None
    for attempt in range(1, retries + 1):
        try:
            resume_at = tmp_path.stat().st_size if tmp_path.exists() and not force else 0
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
            if resume_at:
                headers["Range"] = f"bytes={resume_at}-"

            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as response:
                response.raise_for_status()
                if resume_at and response.status_code != 206:
                    resume_at = 0
                    tmp_path.unlink(missing_ok=True)

                mode = "ab" if resume_at else "wb"
                content_length = response.headers.get("Content-Length")
                total = remote_size
                if total is None and content_length and content_length.isdigit():
                    total = int(content_length) + resume_at
                progress = _progress(
                    total=total,
                    initial=resume_at if total else 0,
                    unit="B",
                    unit_scale=True,
                    desc=output_path.name,
                )
                try:
                    with tmp_path.open(mode) as file:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if not chunk:
                                continue
                            file.write(chunk)
                            if progress is not None:
                                progress.update(len(chunk))
                finally:
                    if progress is not None:
                        progress.close()

            if remote_size is not None and tmp_path.stat().st_size != remote_size:
                raise requests.RequestException(
                    f"Incomplete download for {url}: "
                    f"{tmp_path.stat().st_size} / {remote_size} bytes"
                )
            tmp_path.replace(output_path)
            return output_path
        except requests.RequestException as error:
            last_error = error
            wait_seconds = 5 * attempt
            print(f"Download failed ({attempt}/{retries}): {url} - {error}")
            if attempt < retries:
                time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to download {url}") from last_error


def download_ai4boundaries(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    base_url: str = BASE_URL,
    force: bool = False,
    extract: bool = True,
    remove_archives: bool = False,
    retries: int = 3,
) -> list[Path]:
    """Download AI4Boundaries and extract archives under ``output_dir``."""

    output_dir = Path(output_dir)
    archive_root = output_dir / ARCHIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    print(f"Collecting file URLs from {base_url}")
    file_urls = collect_file_urls(base_url, session=session)
    if not file_urls:
        raise RuntimeError(f"No files found at {base_url}")

    print(f"Found {len(file_urls)} files.")
    downloaded_paths: list[Path] = []
    iterator: Iterable[str]
    progress = _progress(file_urls, desc="Files")
    iterator = progress if progress is not None else file_urls
    for file_url in iterator:
        relative_path = _remote_relative_path(file_url, base_url)
        target_path = (
            archive_root / relative_path if _is_archive(relative_path) else output_dir / relative_path
        )
        downloaded_path = download_file(
            file_url,
            target_path,
            session,
            force=force,
            retries=retries,
        )
        downloaded_paths.append(downloaded_path)

        if extract and _is_archive(downloaded_path):
            extract_dir = _archive_output_dir(downloaded_path, archive_root, output_dir)
            extract_archive(downloaded_path, extract_dir, force=force)
            if remove_archives:
                downloaded_path.unlink(missing_ok=True)

    print(f"AI4Boundaries is ready under: {output_dir}")
    return downloaded_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and unpack AI4Boundaries into datasets/AI4Boundaries."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the dataset will be prepared.",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="AI4Boundaries Apache index URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files and re-extract archives even when outputs exist.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download files; do not extract archives.",
    )
    parser.add_argument(
        "--remove-archives",
        action="store_true",
        help="Delete archive files after successful extraction.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts per file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_ai4boundaries(
        args.output_dir,
        base_url=args.base_url,
        force=args.force,
        extract=not args.no_extract,
        remove_archives=args.remove_archives,
        retries=args.retries,
    )
