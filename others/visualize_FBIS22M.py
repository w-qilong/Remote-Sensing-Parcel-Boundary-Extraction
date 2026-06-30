"""Visualize FBIS-22M images with YOLO polygon labels overlaid."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
DEFAULT_DATA_ROOT = Path("datasets") / "FBIS-22M"
DEFAULT_OUTPUT_DIR = Path("outputs") / "fbis22m_visualizations"
FIGURE_TITLE = "FBIS-22M labels"


@dataclass(frozen=True)
class FBISSample:
    image_path: Path
    label_path: Path
    part: str


def collect_fbis22m_samples(data_root: Path | str = DEFAULT_DATA_ROOT) -> list[FBISSample]:
    """Collect image/label pairs below an FBIS-22M root directory.

    The expected layout is one or more part directories containing sibling
    ``images`` and ``labels`` folders, for example:

    ``datasets/FBIS-22M/FBIS-22M_part-ae/images/*.tif``
    ``datasets/FBIS-22M/FBIS-22M_part-ae/labels/*.txt``
    """

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"FBIS-22M root not found: {root}")

    samples: list[FBISSample] = []
    for images_dir in sorted(root.rglob("images")):
        if not images_dir.is_dir():
            continue

        part_dir = images_dir.parent
        labels_dir = part_dir / "labels"
        if not labels_dir.is_dir():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append(
                    FBISSample(
                        image_path=image_path,
                        label_path=label_path,
                        part=part_dir.name,
                    )
                )

    return samples


def _read_rgb_image(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        band_count = min(src.count, 3)
        image = src.read(list(range(1, band_count + 1))).astype(np.float32)

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    elif image.shape[0] == 2:
        image = np.concatenate([image, image[:1]], axis=0)

    image = np.transpose(image[:3], (1, 2, 0))
    rgb = np.zeros_like(image, dtype=np.uint8)
    for band_index in range(3):
        band = image[:, :, band_index]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue

        low, high = np.percentile(valid, (1, 99))
        if high <= low:
            low, high = float(valid.min()), float(valid.max())
        if high <= low:
            continue

        rgb[:, :, band_index] = np.clip((band - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)

    return rgb


def read_yolo_polygons(label_path: Path | str, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    """Read YOLO segmentation labels and return pixel-space polygons."""

    polygons: list[tuple[int, np.ndarray]] = []
    path = Path(label_path)
    if not path.exists():
        return polygons

    with path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            if (len(parts) - 1) % 2 != 0:
                raise ValueError(f"Invalid polygon coordinate count at {path}:{line_number}")

            class_id = int(float(parts[0]))
            coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
            polygons.append((class_id, np.rint(coords).astype(np.int32)))

    return polygons


def overlay_yolo_polygons(
    image_rgb: np.ndarray,
    polygons: list[tuple[int, np.ndarray]],
    *,
    fill_alpha: float = 0.22,
    line_color: tuple[int, int, int] = (255, 64, 64),
    line_width: int = 1,
) -> np.ndarray:
    """Overlay polygon labels on an RGB image."""

    output = image_rgb.copy()
    overlay = image_rgb.copy()

    for index, (_, polygon) in enumerate(polygons):
        if polygon.shape[0] < 3:
            continue

        color = (
            int((37 * index + 255) % 256),
            int((97 * index + 128) % 256),
            int((173 * index + 64) % 256),
        )
        cv2.fillPoly(overlay, [polygon], color=color)

    output = cv2.addWeighted(overlay, fill_alpha, output, 1.0 - fill_alpha, 0)
    for _, polygon in polygons:
        if polygon.shape[0] < 3:
            continue
        cv2.polylines(output, [polygon], isClosed=True, color=line_color, thickness=line_width, lineType=cv2.LINE_AA)

    return output


def visualize_fbis22m_sample(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    *,
    image_name: str | None = None,
    output_path: Path | str | None = None,
    fill_alpha: float = 0.22,
) -> Path:
    """Read FBIS-22M and save one image with its matching label polygons overlaid."""

    samples = collect_fbis22m_samples(data_root)
    if not samples:
        raise FileNotFoundError(f"No FBIS-22M image/label pairs found under: {data_root}")

    if image_name is None:
        sample = samples[0]
    else:
        sample = next(
            (
                item
                for item in samples
                if item.image_path.name == image_name or item.image_path.stem == Path(image_name).stem
            ),
            None,
        )
        if sample is None:
            raise FileNotFoundError(f"No image/label pair found for image name: {image_name}")

    image = _read_rgb_image(sample.image_path)
    height, width = image.shape[:2]
    polygons = read_yolo_polygons(sample.label_path, width, height)
    visualized = overlay_yolo_polygons(image, polygons, fill_alpha=fill_alpha)

    if output_path is None:
        output_dir = DEFAULT_OUTPUT_DIR
        output_path = output_dir / f"{sample.image_path.stem}_overlay.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
    return output_path


def _find_sample_index(samples: list[FBISSample], image_name: str | None) -> int:
    if image_name is None:
        return 0

    image_stem = Path(image_name).stem
    for index, sample in enumerate(samples):
        if sample.image_path.name == image_name or sample.image_path.stem == image_stem:
            return index

    raise FileNotFoundError(f"No image/label pair found for image name: {image_name}")


def _render_sample(sample: FBISSample, fill_alpha: float) -> tuple[np.ndarray, int]:
    image = _read_rgb_image(sample.image_path)
    height, width = image.shape[:2]
    polygons = read_yolo_polygons(sample.label_path, width, height)
    return overlay_yolo_polygons(image, polygons, fill_alpha=fill_alpha), len(polygons)


def _save_visualization(image_rgb: np.ndarray, output_dir: Path, sample: FBISSample) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample.image_path.stem}_overlay.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return output_path


def browse_fbis22m_samples(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    *,
    image_name: str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    fill_alpha: float = 0.22,
) -> None:
    """Show FBIS-22M label overlays and switch samples with arrow keys.

    Keys:
    - Right arrow, ``d``, or Space: next sample
    - Left arrow or ``a``: previous sample
    - ``q`` or Escape: quit
    """

    samples = collect_fbis22m_samples(data_root)
    if not samples:
        raise FileNotFoundError(f"No FBIS-22M image/label pairs found under: {data_root}")

    output_dir = Path(output_dir)
    state = {"index": _find_sample_index(samples, image_name)}

    fig, ax = plt.subplots(num=FIGURE_TITLE)
    ax.axis("off")
    image_artist = None

    def show_current_sample() -> None:
        nonlocal image_artist
        index = state["index"]
        sample = samples[index]
        visualized, polygon_count = _render_sample(sample, fill_alpha)
        saved_path = _save_visualization(visualized, output_dir, sample)

        title = f"{index + 1}/{len(samples)} {sample.image_path.name} labels={polygon_count}"
        if image_artist is None:
            image_artist = ax.imshow(visualized)
        else:
            image_artist.set_data(visualized)
        ax.set_title(title)
        fig.canvas.draw_idle()
        print(f"Showing {title} | saved: {saved_path}")

    def on_key_press(event) -> None:
        if event.key in {"left", "a"}:
            state["index"] = (state["index"] - 1) % len(samples)
            show_current_sample()
        elif event.key in {"right", "d", " "}:
            state["index"] = (state["index"] + 1) % len(samples)
            show_current_sample()
        elif event.key in {"escape", "q"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    show_current_sample()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse FBIS-22M images with YOLO polygon labels.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to datasets/FBIS-22M.")
    parser.add_argument("--image-name", default=None, help="Initial image filename or stem. Defaults to first pair.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved overlay PNGs.")
    parser.add_argument("--fill-alpha", type=float, default=0.22, help="Polygon fill transparency.")
    args = parser.parse_args()

    browse_fbis22m_samples(
        args.data_root,
        image_name=args.image_name,
        output_dir=args.output_dir,
        fill_alpha=args.fill_alpha,
    )


if __name__ == "__main__":
    main()
