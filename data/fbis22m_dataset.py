"""FBIS-22M instance segmentation dataset.

本模块面向 ``Field-DINO-Mask2Former`` 设计：从 FBIS-22M 的
``images``/``labels`` 目录读取影像和 YOLO polygon 标签，并在运行时生成
实例分割训练所需的 masks、boxes、boundary 和 distance map。

文件名遵循项目动态加载约定：
``fbis22m_dataset.py`` -> ``Fbis22mDataset``。
因此后续可以通过 ``--train_dataset fbis22m_dataset`` 进行加载。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    # rasterio 对 GeoTIFF 支持更完整：可以正确读取多波段、地理栅格元数据等。
    # Dataset 当前只使用像素值本身，但优先用 rasterio 可以减少遥感 TIFF 读取失败。
    import rasterio
except ImportError:  # pragma: no cover - Pillow fallback still supports common TIFFs.
    # 有些轻量测试环境没有 rasterio；这时退回 Pillow 读取普通 RGB/TIFF。
    # 真正训练 FBIS-22M 时仍建议安装 pyproject.toml 中声明的 rasterio。
    rasterio = None


# 可被识别为影像的文件后缀。FBIS-22M 主体是 GeoTIFF，但这里也兼容常见图片格式，
# 方便用户构造小型调试数据集。
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
# 默认数据根目录与项目现有下载脚本、可视化脚本保持一致。
DEFAULT_DATA_ROOT = Path("datasets") / "FBIS-22M"
# parts="all" 表示扫描全部 FBIS-22M_part-* 子目录。
ALL_PARTS = "all"
# split="all" 主要用于 smoke test 或可视化调试；训练时通常用 train/val/test。
SPLIT_NAMES = {"train", "val", "test", "all"}
# DINO/ViT backbone 通常沿用 ImageNet 归一化，这里与 ftw_dataset.py 保持一致。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Fbis22mSample:
    """One FBIS-22M image/label pair.

    这个轻量 dataclass 只保存路径和所属 part，不提前读取影像或标签。
    这样 Dataset 初始化阶段只做文件索引，真正的 IO 与栅格化放在
    ``__getitem__`` 中按需执行，避免大数据集启动时占用大量内存。
    """

    image_path: Path
    label_path: Path
    part: str

    @property
    def stem(self) -> str:
        # stem 用于与用户传入的 file_names 进行匹配，例如 "A.tif" 与 "A" 等价。
        return self.image_path.stem

    @property
    def display_name(self) -> str:
        # 同名影像可能出现在不同 part 中；加 part 前缀后日志和调试输出更明确。
        return f"{self.part}/{self.image_path.name}"


def _resolve_data_root(data_root: str | Path) -> Path:
    """Resolve project-relative FBIS-22M paths in the same style as FtwDataset."""

    root = Path(data_root)
    # 绝对路径或相对当前工作目录已存在时，直接使用用户传入值。
    if root.is_absolute() or root.exists():
        return root

    # 如果用户从项目外部启动训练，当前工作目录可能不是仓库根目录。
    # 这里再尝试按“项目根目录/data/..”推导，提升 CLI 调用的稳健性。
    project_root = Path(__file__).resolve().parents[1]
    project_relative = project_root / root
    if project_relative.exists():
        return project_relative

    return root


def _as_list(value: str | Sequence[str]) -> list[str]:
    """Normalize CLI-friendly single-or-many values to a list."""

    # argparse 的 nargs="+" 会给 list；用户手动实例化 Dataset 时常传字符串。
    # 统一成 list 后，后续 parts 过滤逻辑不用关心输入形式。
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_part_name(part_name: str) -> str:
    """Allow both full part directory names and short suffixes such as ``ae``."""

    part_name = part_name.strip()
    # 用户可以写 "FBIS-22M_part-ae"，也可以简写成 "ae"。
    if part_name.startswith("FBIS-22M_part-"):
        return part_name
    return f"FBIS-22M_part-{part_name}"


def _matches_part(part_dir: Path, parts: Sequence[str]) -> bool:
    """Check whether a discovered part directory should be included."""

    # parts="all" 是训练全量 FBIS-22M 时的默认行为。
    if any(part.lower() == ALL_PARTS for part in parts):
        return True
    # 对所有显式 part 做一次规范化，再与真实目录名比较。
    normalized = {_normalize_part_name(part) for part in parts}
    return part_dir.name in normalized


def collect_fbis22m_instance_samples(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    parts: str | Sequence[str] = ALL_PARTS,
    image_suffixes: set[str] | None = None,
) -> list[Fbis22mSample]:
    """Collect FBIS-22M image/label pairs below a dataset root.

    Expected layout:

    ```text
    datasets/FBIS-22M/FBIS-22M_part-ae/images/*.tif
    datasets/FBIS-22M/FBIS-22M_part-ae/labels/*.txt
    ```
    """

    root = _resolve_data_root(data_root)
    if not root.exists():
        raise FileNotFoundError(f"FBIS-22M root not found: {root}")

    suffixes = image_suffixes or IMAGE_SUFFIXES
    selected_parts = _as_list(parts)
    samples: list[Fbis22mSample] = []

    # 使用 rglob("images") 是为了兼容未来可能更深一层的目录组织；
    # 只要某个 part 下有 sibling labels 目录，就会被识别为 FBIS-22M part。
    for images_dir in sorted(root.rglob("images")):
        if not images_dir.is_dir():
            continue

        part_dir = images_dir.parent
        # part 过滤发生在扫描样本之前，避免在全量数据上做无谓文件遍历。
        if not _matches_part(part_dir, selected_parts):
            continue

        labels_dir = part_dir / "labels"
        if not labels_dir.is_dir():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in suffixes:
                continue
            # FBIS-22M 的影像和 YOLO polygon 标签同 stem：images/A.tif -> labels/A.txt。
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append(
                    Fbis22mSample(
                        image_path=image_path,
                        label_path=label_path,
                        part=part_dir.name,
                    )
                )

    if not samples:
        # 这里尽早报错，比训练跑到第一个 batch 才发现路径或 part 写错更友好。
        raise FileNotFoundError(
            f"No FBIS-22M image/label pairs found under {root} for parts={selected_parts}"
        )

    return samples


def _stable_unit_interval(value: str, seed: int) -> float:
    """Map a sample id to a deterministic number in [0, 1)."""

    # Python 内置 hash 会受 PYTHONHASHSEED 影响，跨进程/机器不稳定。
    # SHA1 虽然不是为了随机数设计，但足够把样本名稳定映射到一个均匀桶。
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _split_samples(
    samples: Sequence[Fbis22mSample],
    split: str,
    split_ratios: tuple[float, float, float],
    split_seed: int,
) -> list[Fbis22mSample]:
    """Create deterministic train/val/test splits when FBIS has no split folders."""

    split = split.lower()
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {sorted(SPLIT_NAMES)}, got {split!r}")
    if split == "all":
        # split="all" 保留全部样本，适合做可视化、debug 或外部已划分好的小数据集。
        return list(samples)

    train_ratio, val_ratio, test_ratio = split_ratios
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

    train_end = train_ratio
    val_end = train_ratio + val_ratio
    selected: list[Fbis22mSample] = []
    for sample in samples:
        # 每个样本根据 display_name 进入固定桶，因此调整 batch_size、num_workers
        # 或文件遍历顺序都不会改变某个样本属于 train/val/test 的结果。
        bucket = _stable_unit_interval(sample.display_name, split_seed)
        if split == "train" and bucket < train_end:
            selected.append(sample)
        elif split == "val" and train_end <= bucket < val_end:
            selected.append(sample)
        elif split == "test" and val_end <= bucket:
            selected.append(sample)

    if not selected:
        raise FileNotFoundError(
            f"No FBIS-22M samples left for split={split!r}. "
            f"Check split_ratios={split_ratios} or use split='all' for debugging."
        )
    return selected


def _read_with_rasterio(image_path: Path) -> np.ndarray:
    """Read geospatial rasters as HWC arrays, preserving up to the first 3 bands."""

    if rasterio is None:
        raise ImportError

    with rasterio.open(image_path) as src:
        # 目前模型输入是 RGB 三通道；多光谱数据先取前三个波段作为 RGB-like 输入。
        # 如果后续要利用更多波段，应同步修改模型 in_channels 和归一化策略。
        band_count = min(src.count, 3)
        image = src.read(list(range(1, band_count + 1)))

    if image.shape[0] == 1:
        # 单波段数据复制成 3 通道，避免破坏 DINO 预训练主干的输入约定。
        image = np.repeat(image, 3, axis=0)
    elif image.shape[0] == 2:
        # 两波段数据用第一个波段补齐第三通道，至少保持形状可训练/可调试。
        image = np.concatenate([image, image[:1]], axis=0)

    # rasterio 返回 CHW；后续 OpenCV/NumPy 处理更习惯 HWC。
    return np.transpose(image[:3], (1, 2, 0))


def _read_with_pillow(image_path: Path) -> np.ndarray:
    """Fallback reader for common RGB images and simple TIFFs."""

    with Image.open(image_path) as image:
        # Pillow 分支强制 RGB，主要服务于普通图片、小型单元测试或无 rasterio 环境。
        return np.asarray(image.convert("RGB"))


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert remote-sensing image arrays to display/training RGB uint8."""

    if image.ndim != 3:
        raise ValueError(f"Expected HWC image array, got shape={image.shape}")

    if image.dtype == np.uint8:
        # 已经是 8-bit RGB/RGBA 时，只保留前三个通道。
        return image[:, :, :3]

    # 遥感影像常见 uint16/float 取值范围，不能简单除以 255。
    # 这里按每个波段的 1-99 百分位拉伸到 uint8，降低极端亮暗像素影响。
    image = image.astype(np.float32, copy=False)
    rgb = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)
    for band_index in range(3):
        band = image[:, :, band_index]
        # 排除 NaN/Inf，避免 percentile 结果污染整幅图。
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue

        low, high = np.percentile(valid, (1, 99))
        if high <= low:
            # 对近似常量图像退回 min/max；若仍无动态范围，则该通道保持 0。
            low, high = float(valid.min()), float(valid.max())
        if high <= low:
            continue

        scaled = (band - low) * 255.0 / (high - low)
        rgb[:, :, band_index] = np.clip(scaled, 0, 255).astype(np.uint8)

    return rgb


def read_rgb_image(image_path: str | Path) -> np.ndarray:
    """Read an FBIS-22M image as RGB uint8 HWC array."""

    path = Path(image_path)
    try:
        # 优先走 rasterio，兼容 GeoTIFF 和多波段遥感数据。
        image = _read_with_rasterio(path)
    except ImportError:
        # 没装 rasterio 时，尽量用 Pillow 跑通普通图片和轻量测试。
        image = _read_with_pillow(path)
    return _to_uint8_rgb(image)


def _parse_resolution_from_name(path: Path) -> float | None:
    """Best-effort resolution parser for names like ``S2_10m`` or ``MAX_120cm``."""

    name = path.stem.lower()
    # 文件名中常见 S2_10m、PL_3m 这类米级分辨率标记。
    meter_match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)m(?![a-z])", name)
    if meter_match:
        return float(meter_match.group(1))

    # MAX_120cm、MAX_30cm 这类厘米单位统一换算成米，便于后续 resolution embedding。
    centimeter_match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)cm(?![a-z])", name)
    if centimeter_match:
        return float(centimeter_match.group(1)) / 100.0

    return None


def read_yolo_polygons(
    label_path: str | Path,
    width: int,
    height: int,
    *,
    min_points: int = 3,
) -> list[tuple[int, np.ndarray]]:
    """Read YOLO segmentation labels and return pixel-space polygons."""

    polygons: list[tuple[int, np.ndarray]] = []
    path = Path(label_path)
    if not path.exists():
        # 缺标签时返回空实例列表，调用方会生成空 masks/boxes。
        # 正常训练数据应保证标签存在；这个分支主要提高调试容错性。
        return polygons

    with path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 1 + min_points * 2:
                # 少于一个类别 id + min_points 个点，无法形成有效多边形，直接跳过。
                continue
            if (len(parts) - 1) % 2 != 0:
                # YOLO segmentation 每个点是 x/y 成对出现；奇数坐标说明标签文件有问题。
                raise ValueError(f"Invalid polygon coordinate count at {path}:{line_number}")

            class_id = int(float(parts[0]))
            # YOLO polygon 坐标是归一化到 [0, 1] 的 x/y；先 reshape 成 N x 2。
            coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
            # 转成像素坐标，并 clamp 到图像边界内，避免 OpenCV 栅格化越界。
            coords[:, 0] = np.clip(coords[:, 0] * width, 0, width - 1)
            coords[:, 1] = np.clip(coords[:, 1] * height, 0, height - 1)
            # cv2.fillPoly 需要 int32 像素点；四舍五入比直接 floor 更少系统性偏移。
            polygon = np.rint(coords).astype(np.int32)
            if polygon.shape[0] >= min_points:
                polygons.append((class_id, polygon))

    return polygons


def _polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize one pixel-space polygon to a binary mask."""

    mask = np.zeros((height, width), dtype=np.uint8)
    # OpenCV 多边形填充速度快，适合每个 batch 动态把 YOLO polygon 转成 mask。
    cv2.fillPoly(mask, [polygon], color=1)
    return mask


def _polygon_to_box(polygon: np.ndarray, width: int, height: int) -> list[float]:
    """Convert one polygon to an xyxy box in pixel coordinates."""

    # boxes 用于后续 Hungarian matching 或可选 box/GIoU 辅助损失。
    x_min = float(np.clip(polygon[:, 0].min(), 0, width - 1))
    y_min = float(np.clip(polygon[:, 1].min(), 0, height - 1))
    x_max = float(np.clip(polygon[:, 0].max(), 0, width - 1))
    y_max = float(np.clip(polygon[:, 1].max(), 0, height - 1))
    return [x_min, y_min, x_max, y_max]


def rasterize_instances(
    polygons: Sequence[tuple[int, np.ndarray]],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize polygons to instance masks, boxes and labels."""

    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    labels: list[int] = []

    for class_id, polygon in polygons:
        # 每一行 YOLO polygon 对应一个田块实例；这里逐实例栅格化。
        mask = _polygon_to_mask(polygon, height, width)
        if mask.sum() == 0:
            # 极小、多边形退化或全在图外的实例会得到空 mask，不参与训练目标。
            continue
        masks.append(mask)
        boxes.append(_polygon_to_box(polygon, width, height))
        labels.append(class_id)

    if not masks:
        # 保持空实例时的张量维度稳定：N=0，但 H/W 和 boxes/labels 形状仍合法。
        # 这对 collate 和后续 matcher 都很重要。
        return (
            np.zeros((0, height, width), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(masks, axis=0).astype(np.uint8),
        np.asarray(boxes, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def make_boundary_map(instance_masks: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Create a binary boundary map from instance masks."""

    if instance_masks.shape[0] == 0:
        return np.zeros(instance_masks.shape[1:], dtype=np.uint8)

    thickness = max(1, int(thickness))
    # 通过 mask - erode(mask) 得到实例内侧边界。对每个实例独立做边界，
    # 可以保留相邻田块之间的共享边界，而不是只保留 union mask 的外轮廓。
    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    boundary = np.zeros(instance_masks.shape[1:], dtype=np.uint8)
    for mask in instance_masks:
        mask_uint8 = mask.astype(np.uint8)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        instance_boundary = mask_uint8 - eroded
        # 多个实例边界取并集，作为 dense boundary auxiliary target。
        boundary = np.maximum(boundary, instance_boundary)
    return boundary.astype(np.uint8)


def make_distance_map(instance_masks: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Create an interior distance transform map for each field instance.

    ``instance_masks`` 推荐传入形状为 ``[N, H, W]`` 的实例二值 mask。函数会对
    每个实例单独计算距离变换，再把结果合并成一张 ``[H, W]`` dense target。
    这样相邻田块不会因为 mask union 被连接成一个大区域，实例共享边附近
    仍会保持靠近边界的低距离值。
    """

    if instance_masks.ndim == 2:
        # 兼容旧调用：二维输入被视为单个实例 mask。FBIS Dataset 内部会传入
        # 三维实例 masks，才能准确保留相邻实例之间的边界语义。
        instance_masks = instance_masks[None, :, :]
    if instance_masks.ndim != 3:
        raise ValueError(f"Expected instance_masks with shape [N, H, W], got {instance_masks.shape}")

    if instance_masks.shape[0] == 0:
        return np.zeros(instance_masks.shape[1:], dtype=np.float32)

    distance_map = np.zeros(instance_masks.shape[1:], dtype=np.float32)
    for mask in instance_masks:
        mask_uint8 = (mask > 0).astype(np.uint8)
        if mask_uint8.max() == 0:
            # 极小退化实例可能在栅格化后为空；跳过可避免无意义的除零和噪声。
            continue

        # distanceTransform 必须在单个实例内部计算。若先把所有实例做 union，
        # 相邻或接触的田块会被误认为同一个连通区域，中心距离会跨实例扩张。
        instance_distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        if normalize and instance_distance.max() > 0:
            # 按实例各自归一化到 [0, 1]，让小田块也能提供足够强的距离监督。
            instance_distance = instance_distance / instance_distance.max()

        # 多个实例合成 dense distance target。重叠像素取较大值，表示它至少位于
        # 某个实例的内部较深处；非重叠区域则天然保留各实例自己的距离形态。
        distance_map = np.maximum(distance_map, instance_distance.astype(np.float32))

    return distance_map.astype(np.float32)


def image_to_tensor(
    image_rgb: np.ndarray,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert uint8 RGB HWC image to normalized CHW tensor."""

    # 模型输入使用 PyTorch 标准 CHW 格式，并把 uint8 [0, 255] 转为 float [0, 1]。
    image = torch.from_numpy(image_rgb.astype(np.float32).transpose(2, 0, 1)) / 255.0
    if normalize:
        # DINO/ViT 预训练通常使用 ImageNet 归一化，保持分布一致有助于迁移。
        mean_tensor = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
        std_tensor = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
        image = (image - mean_tensor) / std_tensor
    return image


def _normalize_resize_size(resize_size: int | Sequence[int] | None) -> tuple[int, int] | None:
    """Return resize size as ``(height, width)`` or None."""

    if resize_size is None:
        # None 表示保留 FBIS-22M 原始 patch 尺寸，例如 256 或 512。
        return None
    if isinstance(resize_size, int):
        if resize_size <= 0:
            raise ValueError("resize_size must be positive.")
        # 单个 int 视为正方形 resize，训练时常用 resize_size=512。
        return (resize_size, resize_size)

    values = list(resize_size)
    if len(values) != 2:
        raise ValueError("resize_size must be an int or a two-value sequence: (height, width).")
    height, width = int(values[0]), int(values[1])
    if height <= 0 or width <= 0:
        raise ValueError("resize_size values must be positive.")
    return (height, width)


def _resize_image(image_rgb: np.ndarray, resize_size: tuple[int, int] | None) -> np.ndarray:
    """Resize RGB image before label rasterization."""

    if resize_size is None:
        return image_rgb
    height, width = resize_size
    # 注意：这里先 resize image，再按新 H/W 从归一化 polygon 生成标签。
    # 因为 YOLO 坐标是归一化坐标，mask/box 会天然对齐 resize 后的尺寸。
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)


class Fbis22mDataset(Dataset):
    """FBIS-22M dataset for query-based field instance segmentation.

    ``__getitem__`` 返回字典，便于后续与 Mask2Former/DETR 风格训练代码对接。
    由于每张图的实例数量不同，使用 DataLoader 时应传入
    ``collate_fn=Fbis22mDataset.collate_fn``，或让 DataModule 自动读取
    该静态方法。

    返回的关键字段：
    - ``image``: 归一化后的影像张量，形状为 ``[3, H, W]``。
    - ``instances``: 实例级标注字典，包含 ``masks``、``boxes``、``labels``。
    - ``boundary``: 实例边界并集，用于边界辅助损失。
    - ``distance``: 前景内部距离变换，用于几何辅助损失。
    """

    def __init__(
        self,
        data_root: str = str(DEFAULT_DATA_ROOT),
        parts: str | Sequence[str] = ALL_PARTS,
        split: str = "train",
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 1234,
        file_names: Sequence[str] | None = None,
        max_samples: int | None = None,
        normalize_image: bool = True,
        resize_size: int | Sequence[int] | None = None,
        boundary_thickness: int = 3,
        return_polygons: bool = False,
        return_image_path: bool = True,
        min_polygon_points: int = 3,
    ) -> None:
        # 保存配置，便于训练日志、调试和后续访问。
        self.data_root = _resolve_data_root(data_root)
        self.parts = _as_list(parts)
        self.split = split
        self.split_ratios = split_ratios
        self.split_seed = split_seed
        self.normalize_image = normalize_image
        self.resize_size = _normalize_resize_size(resize_size)
        self.boundary_thickness = boundary_thickness
        self.return_polygons = return_polygons
        self.return_image_path = return_image_path
        self.min_polygon_points = min_polygon_points

        # 第一步只建立样本索引：找到所有 image/label 对，不读取大影像。
        samples = collect_fbis22m_instance_samples(self.data_root, self.parts)
        if file_names is not None:
            # file_names 既支持 stem，也支持 display_name，便于用户用可视化脚本里看到的名字筛样本。
            wanted = {Path(name).stem for name in file_names}
            samples = [sample for sample in samples if sample.stem in wanted or sample.display_name in wanted]
            if not samples:
                raise FileNotFoundError(f"No FBIS-22M samples matched file_names={list(file_names)}")

        # 如果数据集没有官方 split 文件，这里用样本名哈希做稳定划分。
        self.samples = _split_samples(samples, split, split_ratios, split_seed)
        if max_samples is not None:
            # max_samples 常用于 smoke test 或快速检查 Dataset 输出形状。
            self.samples = self.samples[:max_samples]

        # 与 FtwDataset 保持类似：暴露一个可读的样本名列表，方便测试和调试。
        self.file_names = [sample.display_name for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        # 1. 读取影像并可选 resize。影像最终是 HWC uint8 RGB。
        image_rgb = read_rgb_image(sample.image_path)
        image_rgb = _resize_image(image_rgb, self.resize_size)
        height, width = image_rgb.shape[:2]

        # 2. 根据当前影像尺寸，把 YOLO 归一化 polygon 转成像素坐标。
        polygons = read_yolo_polygons(
            sample.label_path,
            width,
            height,
            min_points=self.min_polygon_points,
        )

        # 3. 从 polygon 生成实例分割监督：masks/boxes/labels。
        masks_np, boxes_np, labels_np = rasterize_instances(polygons, height, width)

        # 4. 生成辅助监督。模型设计中不再预测语义分割 mask，因此只保留
        # boundary/distance 两个 dense target；二者都按实例内部逻辑生成。
        boundary_np = make_boundary_map(masks_np, thickness=self.boundary_thickness)
        distance_np = make_distance_map(masks_np)

        # DETR/Mask2Former 训练通常让每张图保留自己的 instances dict；
        # batch 维度由 collate_fn 组织，不能在这里强行 stack 不同 N 的实例。
        instances = {
            "masks": torch.from_numpy(masks_np).float(),
            "boxes": torch.from_numpy(boxes_np).float(),
            "labels": torch.from_numpy(labels_np).long(),
        }

        # 主返回值采用字典格式，比 tuple 更适合实例分割这种多监督任务。
        # 之后的 LightningModule 可以按 key 取需要的监督信号。
        item: dict[str, object] = {
            "file_name": sample.display_name,
            "image": image_to_tensor(image_rgb, normalize=self.normalize_image),
            "instances": instances,
            "boundary": torch.from_numpy(boundary_np[None, :, :]).long(),
            "distance": torch.from_numpy(distance_np[None, :, :]).float(),
            "height": height,
            "width": width,
            "resolution": _parse_resolution_from_name(sample.image_path),
            "part": sample.part,
        }

        if self.return_polygons:
            # polygon 原始点可用于可视化、debug 或后续做 polygon-level augmentation 检查。
            item["polygons"] = [
                {"label": class_id, "points": torch.from_numpy(polygon).long()}
                for class_id, polygon in polygons
            ]
        if self.return_image_path:
            # 保留路径能让错误样本定位更容易；训练中不需要时可关闭以减小 batch 元信息。
            item["image_path"] = str(sample.image_path)
            item["label_path"] = str(sample.label_path)

        return item

    @staticmethod
    def collate_fn(batch: Sequence[dict[str, object]]) -> dict[str, object]:
        """Collate variable-length instance annotations.

        Images and single-channel dense auxiliary targets are stacked when they have
        equal spatial shape. Instance dictionaries remain as a list because each image
        can contain a different number of instances.
        """

        if not batch:
            raise ValueError("Cannot collate an empty FBIS-22M batch.")

        # 如果用户设置 resize_size，或者 batch 恰好由同尺寸样本组成，就可以直接 stack。
        # 若混合 256/512 原图，则保持 list，后续模型或训练循环可自行 pad/resize。
        image_shapes = {tuple(item["image"].shape) for item in batch}  # type: ignore[index, union-attr]
        can_stack_images = len(image_shapes) == 1
        dense_keys = ("boundary", "distance")

        # 变长实例标注保持 list[dict]，这是 DETR/Mask2Former 常见 batch 约定。
        output: dict[str, object] = {
            "file_name": [item["file_name"] for item in batch],
            "instances": [item["instances"] for item in batch],
            "height": torch.tensor([item["height"] for item in batch], dtype=torch.long),
            "width": torch.tensor([item["width"] for item in batch], dtype=torch.long),
            "resolution": [item["resolution"] for item in batch],
            "part": [item["part"] for item in batch],
        }

        if can_stack_images:
            # 同尺寸情况下，image 和 dense targets 变成标准 B,C,H,W 张量。
            output["image"] = torch.stack([item["image"] for item in batch])  # type: ignore[list-item]
            for key in dense_keys:
                output[key] = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
        else:
            # 不同尺寸保留 list，避免默认 collate 报错，也避免隐式 resize 改变标签。
            output["image"] = [item["image"] for item in batch]
            for key in dense_keys:
                output[key] = [item[key] for item in batch]

        # 可选字段只在 Dataset 配置开启时出现；collate 时保持与 batch 对齐的 list。
        optional_keys = ("image_path", "label_path", "polygons")
        for key in optional_keys:
            if key in batch[0]:
                output[key] = [item.get(key) for item in batch]

        return output


# Alias with the dataset's canonical acronym spelling for direct imports.
FBIS22MInstanceDataset = Fbis22mDataset


if __name__ == "__main__":
    # 轻量手动检查入口：运行本文件会读取一个样本并打印核心张量形状。
    # 完整训练仍应通过 data.DInterface 和 main.py 启动。
    import matplotlib.pyplot as plt

    dataset = FBIS22MInstanceDataset(split="all", max_samples=1, return_polygons=True)
    sample = dataset[0]
    instances = sample["instances"]

    masks = instances["masks"]  # type: ignore[index]
    boxes = instances["boxes"]  # type: ignore[index]
    labels = instances["labels"]  # type: ignore[index]
    boundary = sample["boundary"]
    distance = sample["distance"]

    print("file:", sample["file_name"])
    print("image:", tuple(sample["image"].shape))
    print("instances:", int(masks.shape[0]))
    print("masks:", tuple(masks.shape), masks.dtype)
    print("mask areas:", masks.flatten(1).sum(dim=1).tolist())
    print("boxes xyxy:\n", boxes)
    print("labels:", labels.tolist())
    print("boundary:", tuple(boundary.shape), boundary.dtype)
    print("distance:", tuple(distance.shape), distance.dtype)

    # 反归一化影像，便于 matplotlib 显示真实 RGB 观感。
    image = sample["image"].detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    image_np = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    masks_np = masks.detach().cpu().numpy()
    boundary_np = boundary.squeeze(0).detach().cpu().numpy()
    distance_np = distance.squeeze(0).detach().cpu().numpy()

    # 把 N 个二值 instance mask 合成一张 instance id map：
    # 0 是背景，1..N 分别表示第几个实例，适合快速检查实例是否彼此分离。
    instance_id_map = np.zeros(masks_np.shape[1:], dtype=np.int32)
    for instance_index, mask in enumerate(masks_np, start=1):
        instance_id_map[mask > 0] = instance_index

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Image")

    axes[0, 1].imshow(instance_id_map, cmap="tab20")
    axes[0, 1].set_title("Instance masks")

    axes[0, 2].imshow(image_np)
    axes[0, 2].imshow(instance_id_map, cmap="tab20", alpha=0.45)
    axes[0, 2].set_title("Image + instance masks")

    axes[1, 0].imshow(image_np)
    axes[1, 0].imshow(boundary_np, cmap="gray", alpha=0.65)
    axes[1, 0].set_title("Image + boundary")

    axes[1, 1].imshow(boundary_np, cmap="gray")
    axes[1, 1].set_title("boundary")

    axes[1, 2].imshow(distance_np, cmap="magma")
    axes[1, 2].set_title("distance")

    for axis in axes.ravel():
        axis.axis("off")

    plt.tight_layout()
    plt.show()
