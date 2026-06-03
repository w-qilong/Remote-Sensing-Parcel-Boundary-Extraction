"""Download and preprocess FTW data for HBGNet.

This module is intentionally self-contained so FTW preparation lives with the
HBGNet data layer instead of a separate scripts package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import urllib.request
import sys
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable
import numpy as np

FTW_ARCHIVE_BASE_URL = "https://data.source.coop/kerner-lab/fields-of-the-world-archive"

FTW_CONFIG_DEFAULTS = {
    "download_root": Path("ftw_data") / "ftw_origin_data",
    "ftw_root": None,
    "output_root": Path("ftw_data") / "ftw_dataset",
    "countries": ["rwanda",'kenya'], # 修改此处以包含更多国家，例如 ["rwanda", "malawi", "nigeria"]
    "splits": "train,val,test",
    "max_samples_per_split": None,
    "image_window": "window_b",
    "reflectance_max": 3000.0,
    "boundary_kernel_size": 3,
    "clean_download": False,
}


# =========================
# 一、数据下载与解压阶段
# =========================


def _default_urlretrieve(url: str, path: Path) -> None:
    """默认下载函数；测试中可注入假的 urlretrieve 避免访问网络。"""

    # Source Cooperative 会拒绝部分默认 Python UA，因此这里显式伪装成
    # 常规浏览器请求，降低被服务端拦截的概率。
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request) as response, path.open("wb") as file:
        shutil.copyfileobj(response, file)


def _parse_countries(countries: str | list[str] | tuple[str, ...]) -> list[str]:
    """把逗号分隔字符串或列表统一整理成小写国家列表。"""

    # 允许命令行或配置文件传入两种常见格式："a,b,c" 或 ["a", "b", "c"]。
    if isinstance(countries, str):
        values = countries.split(",")
    else:
        values = countries
    # 去除首尾空格并统一转成小写，确保后续拼接文件名时一致。
    parsed = [country.strip().lower() for country in values if country.strip()]
    if not parsed:
        raise ValueError("At least one FTW country must be provided.")
    return parsed


def normalize_config(config: dict) -> dict:
    """规范化国家列表和 split 列表，便于后续循环处理。"""

    # 先深拷贝一份，避免直接修改传入字典或全局默认配置。
    config = deepcopy(config)
    if not config["countries"]:
        raise ValueError('Set at least one country in FTW_CONFIG_DEFAULTS["countries"].')
    config["countries"] = _parse_countries(config["countries"])
    # split 既支持字符串，也支持列表；这里统一整理为字符串列表。
    if isinstance(config["splits"], str):
        config["splits"] = [
            split.strip() for split in config["splits"].split(",") if split.strip()
        ]
    else:
        config["splits"] = [split.strip() for split in config["splits"] if split.strip()]
    return config


def _load_checksums(path: Path) -> dict[str, str]:
    """读取 Source Cooperative 提供的 checksum.md5 文件。"""

    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            # 文件内容通常是 "country,md5" 这种一行一个国家的格式。
            country, checksum = line.strip().split(",", maxsplit=1)
            checksums[country.lower()] = checksum
    return checksums


def _file_md5(path: Path) -> str:
    """分块计算文件 MD5，避免大 zip 一次性读入内存。"""

    digest = hashlib.md5()
    with path.open("rb") as file:
        # 以 1MB 为块读取，既能减少内存峰值，也不会明显拖慢校验速度。
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unpack_country_archive(zip_path: Path, ftw_root: Path) -> Path:
    """将单个国家 zip 解压到 ftw_root/<country>。"""

    country = zip_path.stem.lower()
    country_root = ftw_root / country
    # 每个国家单独放到自己的目录里，便于后续按国家定位 chips 和影像文件。
    country_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(country_root)
    return country_root


def download_ftw_dataset(
    out_dir: str | Path,
    countries: str | list[str] | tuple[str, ...],
    *,
    base_url: str = FTW_ARCHIVE_BASE_URL,
    clean_download: bool = False,
    unpack: bool = True,
    verify_checksum: bool = True,
    urlretrieve_fn: Callable[[str, Path], None] = _default_urlretrieve,
) -> dict[str, Path]:
    """下载 FTW 的国家压缩包，并按国家解压到指定目录。

    返回值是一个映射：国家名 -> 解压后的国家目录；如果 ``unpack=False``，
    则返回国家名 -> zip 文件路径。
    """
    out_dir = Path(out_dir)
    # clean_download=True 时会先清掉旧下载目录，避免混入过期 zip 或解压结果。
    if clean_download and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source Cooperative 同时提供 checksum.md5 和按国家拆分的 zip 归档。
    base_url = base_url.rstrip("/")
    checksum_path = out_dir / "checksum.md5"
    # 校验文件只需要下载一次；已经存在时直接复用，减少重复网络请求。
    if not checksum_path.exists():
        urlretrieve_fn(f"{base_url}/checksum.md5", checksum_path)
    checksums = _load_checksums(checksum_path)

    results: dict[str, Path] = {}
    ftw_root = out_dir / "ftw"
    for country in _parse_countries(countries):
        zip_path = out_dir / f"{country}.zip"
        # zip 文件按国家命名，缺失时才下载；这样断点重跑时可以直接复用已有文件。
        if not zip_path.exists():
            urlretrieve_fn(f"{base_url}/{country}.zip", zip_path)

        expected_checksum = checksums.get(country)
        # 只有当 checksum 文件里存在对应国家条目时才校验，避免未知国家报错过早。
        if verify_checksum and expected_checksum and _file_md5(zip_path) != expected_checksum:
            raise ValueError(f"Checksum verification failed for {zip_path}")

        # unpack=True 时返回解压后的目录，供后续转换逻辑直接读取。
        results[country] = _unpack_country_archive(zip_path, ftw_root) if unpack else zip_path
    return results


# =========================
# 二、预处理与导出阶段
# =========================


def _normalize_rgb(image: np.ndarray, reflectance_max: float) -> np.ndarray:
    import numpy as np

    # FTW Sentinel-2 影像为多波段反射率；这里仅取前三个波段作为 RGB 输入。
    # 先裁剪到经验上的最大反射率，再缩放到 0~255，便于模型训练和可视化。
    rgb = image[:3].astype(np.float32)
    rgb = np.clip(rgb, 0, reflectance_max) / reflectance_max
    return np.rint(rgb * 255).astype(np.uint8)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    import numpy as np

    # FTW 原始标签通常是类别 id；此处统一转成 0/255 二值掩膜，
    # 让后续 boundary 和 distance 计算都以“是否为目标地块”为准。
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _boundary_from_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    import cv2
    import numpy as np

    # 形态学梯度会保留前景边缘：膨胀与腐蚀的差值就是边界区域。
    # 这个结果可以直接作为边界监督信号。
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def _distance_from_mask(mask: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

    # 距离变换会计算每个前景像素到最近背景像素的距离。
    # 这里再做一次归一化，把结果压缩到 0~255，作为密集监督图使用。
    binary = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    return np.rint(dist * 255).astype(np.uint8)


def _write_tif(path: Path, data: np.ndarray, src_profile: dict) -> None:
    """沿用原始影像 profile 写出 GeoTIFF，更新通道数和数据类型。"""

    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    count = data.shape[0] if data.ndim == 3 else 1
    profile = src_profile.copy()
    # 输出统一使用 uint8；这里保留原始空间参考信息，只修改与数据本身相关的字段。
    profile.update(
        driver="GTiff",
        count=count,
        dtype=rasterio.uint8,
        compress="deflate",
        nodata=None,
    )
    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 3:
            dst.write(data)
        else:
            dst.write(data, 1)


def _iter_split_rows(country_root: Path, country: str, split: str):
    """按 split 遍历 chips parquet 中的样本 id。"""

    import pandas as pd

    chips_path = country_root / f"chips_{country}.parquet"
    chips = pd.read_parquet(chips_path)
    # chips 表里通常记录了每个 aoi_id 属于哪个 split；这里只筛选目标 split。
    split_rows = chips[chips["split"] == split]
    for _, row in split_rows.iterrows():
        yield str(row["aoi_id"])


def _resolve_country_root(ftw_root: Path, country: str) -> Path:
    # 兼容 data/ftw/rwanda 和直接解压到某个国家目录两种布局。
    # 这样无论下载脚本是否重复解压，后续转换都能找到 chips 文件。
    nested_root = ftw_root / country
    if (nested_root / f"chips_{country}.parquet").exists():
        return nested_root
    if (ftw_root / f"chips_{country}.parquet").exists():
        return ftw_root
    raise FileNotFoundError(
        f"Could not find chips_{country}.parquet in {nested_root} or {ftw_root}"
    )


def convert_ftw_to_hbgnet(
    ftw_root: str | Path,
    output_root: str | Path,
    countries: list[str],
    splits: list[str] | None = None,
    max_samples_per_split: int | None = None,
    image_window: str = "window_b",
    reflectance_max: float = 3000.0,
    boundary_kernel_size: int = 3,
) -> dict[str, int]:
    """读取解压后的 FTW 数据，并导出 HBGNet 需要的预处理结果。

    最终目录结构按国家划分：
    output_root/<country>/<split>/image|mask|boundary|dist/<sample_id>.tif
    这样每个国家的数据都独立放在自己的文件夹里，便于后续按国家管理。
    """

    ftw_root = Path(ftw_root)
    output_root = Path(output_root)
    # 默认遍历 train/val/test，除非调用者显式传入其他 split 列表。
    splits = splits or ["train", "val", "test"]
    stats: dict[str, int] = defaultdict(int)

    import rasterio

    for country in _parse_countries(countries):
        country_root = _resolve_country_root(ftw_root, country)
        country_output_root = output_root / country
        for split in splits:
            split_count = 0
            # chips_<country>.parquet 是 FTW 的样本索引，记录 aoi_id 与 split。
            for sample_id in _iter_split_rows(country_root, country, split):
                if max_samples_per_split is not None and split_count >= max_samples_per_split:
                    break

                # 原始 FTW 中影像和标签分别放在不同子目录；这里按 sample_id 配对读取。
                image_path = country_root / "s2_images" / image_window / f"{sample_id}.tif"
                mask_path = country_root / "label_masks" / "semantic_2class" / f"{sample_id}.tif"
                if not image_path.exists() or not mask_path.exists():
                    # 某些样本可能缺少影像或标签，直接跳过，避免整个转换中断。
                    continue

                out_name = f"{sample_id}.tif"
                split_root = country_output_root / split
                with rasterio.open(image_path) as src:
                    image = src.read()
                    image_profile = src.profile
                with rasterio.open(mask_path) as src:
                    mask = _binary_mask(src.read(1))
                    mask_profile = src.profile

                rgb = _normalize_rgb(image, reflectance_max=reflectance_max)
                boundary = _boundary_from_mask(mask, kernel_size=boundary_kernel_size)
                dist = _distance_from_mask(mask)

                # 每个 split 下单独分出 image/mask/boundary/dist 四类结果，方便后续读取。
                _write_tif(split_root / "image" / out_name, rgb, image_profile)
                _write_tif(split_root / "mask" / out_name, mask, mask_profile)
                _write_tif(split_root / "boundary" / out_name, boundary, mask_profile)
                _write_tif(split_root / "dist" / out_name, dist, mask_profile)
                split_count += 1
                stats[split] += 1

    return dict(stats)


def _bootstrap_local_venv_if_needed() -> None:
    """如果当前解释器缺少项目依赖，优先切回仓库内的 .venv。"""

    if importlib.util.find_spec("rasterio") is not None:
        return

    project_root = Path(__file__).resolve().parents[1]
    # Windows 与类 Unix 的虚拟环境入口路径不同，这里分别处理。
    if os.name == "nt":
        candidate = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = project_root / ".venv" / "bin" / "python"

    if not candidate.exists():
        # 如果本地虚拟环境都不存在，就明确提示用户先安装依赖。
        raise ModuleNotFoundError(
            "No module named 'rasterio'. Install project dependencies or run "
            f"with {candidate} if that virtual environment is available."
        )

    if Path(sys.executable).resolve() == candidate.resolve():
        # 已经在本地虚拟环境里，却仍然找不到 rasterio，说明环境本身缺包。
        raise ModuleNotFoundError(
            "No module named 'rasterio'. The local virtual environment exists, "
            "but it does not provide rasterio. Reinstall the project dependencies."
        )

    # 重新启动当前脚本，并显式切换到本地解释器，以便加载对应环境中的依赖。
    os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])


def main(config: dict | None = None) -> None:
    """脚本入口：用字典配置串起下载阶段和预处理阶段。"""

    config = normalize_config(config or FTW_CONFIG_DEFAULTS)
    if config["ftw_root"]:
        ftw_root = Path(config["ftw_root"])
    else:
        # 如果没有提供原始数据目录，就先下载并解压，再进入预处理阶段。
        download_ftw_dataset(
            out_dir=config["download_root"],
            countries=config["countries"],
            clean_download=config["clean_download"],
            unpack=True,
        )
        ftw_root = Path(config["download_root"]) / "ftw"

    stats = convert_ftw_to_hbgnet(
        ftw_root=ftw_root,
        output_root=config["output_root"],
        countries=config["countries"],
        splits=config["splits"],
        max_samples_per_split=config["max_samples_per_split"],
        image_window=config["image_window"],
        reflectance_max=config["reflectance_max"],
        boundary_kernel_size=config["boundary_kernel_size"],
    )
    print(f"Converted samples: {stats}")


if __name__ == "__main__":
    # 先确保当前运行环境具备 rasterio，再执行主流程。
    _bootstrap_local_venv_if_needed()
    main()
