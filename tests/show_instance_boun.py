"""交互浏览 FTW 原始影像与实例标注。

该脚本会扫描 ``ftw_data/ftw_origin_data/ftw`` 下各国家目录中同名的
``s2_images/window_a``、``s2_images/window_b`` 和 ``label_masks/instance``
TIFF 文件，并通过键盘左右方向键切换样本。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np


DEFAULT_FTW_ROOT = Path("ftw_data/ftw_origin_data/ftw")
IMAGE_SCALE = 3000.0


@dataclass(frozen=True)
class FtwInstanceSample:
    """FTW 中一组同名的 window_a、window_b 和 instance 标注。"""

    country: str
    name: str
    window_a: Path
    window_b: Path
    instance: Path


def find_ftw_instance_samples(root: str | Path = DEFAULT_FTW_ROOT) -> list[FtwInstanceSample]:
    """查找 FTW 根目录下所有三类文件都存在的同名样本。

    目录结构要求如下：

    - ``<country>/s2_images/window_a/<name>.tif``
    - ``<country>/s2_images/window_b/<name>.tif``
    - ``<country>/label_masks/instance/<name>.tif``
    """

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"FTW root not found: {root}")

    samples: list[FtwInstanceSample] = []
    for country_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        window_a_dir = country_dir / "s2_images" / "window_a"
        window_b_dir = country_dir / "s2_images" / "window_b"
        instance_dir = country_dir / "label_masks" / "instance"
        if not (window_a_dir.is_dir() and window_b_dir.is_dir() and instance_dir.is_dir()):
            continue

        window_a_names = _tif_names(window_a_dir)
        window_b_names = _tif_names(window_b_dir)
        instance_names = _tif_names(instance_dir)
        common_names = sorted(window_a_names & window_b_names & instance_names)

        samples.extend(
            FtwInstanceSample(
                country=country_dir.name,
                name=name,
                window_a=window_a_dir / name,
                window_b=window_b_dir / name,
                instance=instance_dir / name,
            )
            for name in common_names
        )

    if not samples:
        raise FileNotFoundError(f"No matched FTW samples found under: {root}")
    return samples


def load_ftw_instance_sample(sample: FtwInstanceSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取一个 FTW 样本，返回 window_a、window_b 和实例标注彩色图。"""

    window_a = _read_rgb_tif(sample.window_a)
    window_b = _read_rgb_tif(sample.window_b)
    instance = _read_first_band_tif(sample.instance)
    instance_rgb = colorize_instance_mask(instance)
    return window_a, window_b, instance_rgb


def browse_ftw_instance_samples(
    root: str | Path = DEFAULT_FTW_ROOT,
    start_index: int = 0,
    countries: Iterable[str] | None = None,
    include_empty: bool = False,
) -> None:
    """交互式浏览 FTW 同名样本。

    默认会跳过实例标注全为 0 的样本，避免 Instance 视图一直显示黑图。

    键盘操作：
    - 左方向键 / ``a``：上一张
    - 右方向键 / ``d`` / 空格：下一张
    - ``q`` / ``escape``：关闭窗口
    """

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for interactive browsing. "
            "Please run this script in the project environment with matplotlib installed."
        ) from exc

    samples = find_ftw_instance_samples(root)
    if countries is not None:
        allowed = {country.lower() for country in countries}
        samples = [sample for sample in samples if sample.country.lower() in allowed]
        if not samples:
            raise FileNotFoundError(f"No matched FTW samples found for countries: {sorted(allowed)}")

    nonempty_cache: dict[int, bool] = {}

    def has_visible_instance(index: int) -> bool:
        if include_empty:
            return True
        if index not in nonempty_cache:
            nonempty_cache[index] = has_instance_annotation(samples[index])
        return nonempty_cache[index]

    def find_visible_index(start: int, step: int) -> int:
        for offset in range(len(samples)):
            index = (start + offset * step) % len(samples)
            if has_visible_instance(index):
                return index
        raise FileNotFoundError("No non-empty instance annotations found. Use --include-empty to browse all samples.")

    current_index = start_index % len(samples)
    current_index = find_visible_index(current_index, 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("FTW instance browser")

    def render(index: int) -> None:
        sample = samples[index]
        window_a, window_b, instance_rgb = load_ftw_instance_sample(sample)
        views = (
            ("Window A", window_a),
            ("Window B", window_b),
            ("Instance", instance_rgb),
        )

        for ax, (title, image) in zip(axes, views):
            ax.clear()
            ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"{index + 1}/{len(samples)}  {sample.country}/{sample.name}")
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal current_index
        if event.key in {"right", "d", " "}:
            current_index = find_visible_index((current_index + 1) % len(samples), 1)
            render(current_index)
        elif event.key in {"left", "a"}:
            current_index = find_visible_index((current_index - 1) % len(samples), -1)
            render(current_index)
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render(current_index)
    plt.show()


def colorize_instance_mask(mask: np.ndarray) -> np.ndarray:
    """把实例 ID 映射为稳定 RGB 颜色，背景 0 保持黑色。"""

    import numpy as np

    labels = mask.astype(np.uint64, copy=False)
    red = ((labels * 37 + 17) % 255).astype(np.float32) / 255.0
    green = ((labels * 17 + 83) % 255).astype(np.float32) / 255.0
    blue = ((labels * 29 + 131) % 255).astype(np.float32) / 255.0
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[mask == 0] = 0.0
    return rgb


def has_instance_annotation(sample: FtwInstanceSample) -> bool:
    """判断实例标注中是否存在非背景实例。"""

    return bool(_read_first_band_tif(sample.instance).max() > 0)


def _tif_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.tif")}


def _read_rgb_tif(path: Path) -> np.ndarray:
    import numpy as np

    with _open_tif(path) as src:
        band_count = min(3, src.count)
        image = src.read(list(range(1, band_count + 1)))

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    image = np.moveaxis(image[:3], 0, -1).astype(np.float32) / IMAGE_SCALE
    return np.clip(image, 0.0, 1.0)


def _read_first_band_tif(path: Path) -> np.ndarray:
    with _open_tif(path) as src:
        return src.read(1)


def _open_tif(path: Path):
    try:
        import rasterio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rasterio is required to read FTW TIFF files. "
            "Please run this script in the project environment with rasterio installed."
        ) from exc

    return rasterio.open(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse matched FTW window_a/window_b/instance samples.")
    parser.add_argument("--root", type=Path, default=DEFAULT_FTW_ROOT, help="FTW root directory.")
    parser.add_argument("--start-index", type=int, default=0, help="Initial sample index, zero based.")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include samples whose instance mask is all background.",
    )
    parser.add_argument(
        "--country",
        action="append",
        dest="countries",
        help="Filter by country folder name. Can be used multiple times.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    browse_ftw_instance_samples(
        root=args.root,
        start_index=args.start_index,
        countries=args.countries,
        include_empty=args.include_empty,
    )
