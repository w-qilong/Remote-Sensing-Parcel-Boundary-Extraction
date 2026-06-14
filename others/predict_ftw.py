"""Run FTW test-set prediction and save visual comparison images.

Each saved PNG contains two rows:
1. original RGB image, predicted mask, predicted boundary, predicted distance
2. original RGB image, target mask, target boundary, target distance
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ftw_dataset import FtwDataset
from model import MInterface


# FTW 数据集读取时会使用 ImageNet 均值/方差归一化，以匹配模型主干的输入分布。
# 可视化时需要把张量反归一化回 [0, 255] 的 RGB 图，否则保存出来会颜色失真。
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# 预测脚本不使用 argparse；直接在这里修改配置即可运行。
# checkpoint 是唯一必填项：它应指向训练过程中 ModelCheckpoint 保存的 .ckpt 文件。
# country 支持单国家、多国家或 "all"，规则与 FtwDataset 保持一致。
PREDICT_CONFIG_DEFAULTS = {
    "checkpoint": 'logs/hbg_net_ftw/version_0/checkpoints/best-07.ckpt',
    "data_root": "ftw_data/ftw_dataset",
    "country": ["austria"],
    "split": "test",
    "output_dir": Path("predictions") / "ftw_test",
    "batch_size": 1,
    "num_workers": 0,
    "max_samples": None,
    "mask_threshold": 0.5,
    "device": "auto",
    "return_aux_outputs": True,
}


def normalize_config(config: dict | None = None) -> dict:
    """Merge user config with defaults and normalize path-like values."""

    # deepcopy 避免调用方传入配置后意外修改全局默认字典里的 Path/list 等对象。
    normalized = deepcopy(PREDICT_CONFIG_DEFAULTS)
    if config:
        # 允许外部只覆盖少量参数，例如 main({"max_samples": 20})。
        normalized.update(config)

    if not normalized["checkpoint"]:
        raise ValueError('Set PREDICT_CONFIG_DEFAULTS["checkpoint"] to a .ckpt file path.')

    # 统一转成 Path，后续 mkdir、拼接输出文件名时更稳。
    normalized["checkpoint"] = Path(normalized["checkpoint"])
    normalized["output_dir"] = Path(normalized["output_dir"])
    return normalized


def _safe_name(name: str) -> str:
    """Convert a dataset display name into a filesystem-safe file stem."""

    # 多国家数据集的样本名可能是 "austria/xxx"；直接作为文件名会被解释成子目录。
    return name.replace("/", "__").replace("\\", "__")


def tensor_to_rgb_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW image tensor back to uint8 HWC RGB."""

    # DataLoader 给出的 image 是 CHW，并且已经经过 Normalize。
    # 这里先反归一化，再转成 PIL/常见图像库使用的 HWC。
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    return np.rint(image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def mask_from_logits(mask_logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert model mask logits into a binary mask tensor."""

    if mask_logits.ndim != 4:
        raise ValueError(f"Expected mask logits with shape [B, C, H, W], got {mask_logits.shape}")
    # HBGNet 当前主掩膜分支输出通常是 [B, 1, H, W] logits，需要 sigmoid 后阈值化。
    if mask_logits.size(1) == 1:
        return torch.sigmoid(mask_logits) > threshold
    # 如果未来换成多类 mask 输出，则用 softmax 取非背景类别作为前景。
    return mask_logits.softmax(dim=1).argmax(dim=1, keepdim=True) > 0


def boundary_from_output_or_mask(output, predicted_mask: torch.Tensor) -> torch.Tensor:
    """Use model boundary output when available, otherwise derive edges from mask."""

    # HBGNet return_aux_outputs=True 时输出 [mask_logits, edge_log_probs, distance_map]。
    # edge_log_probs 是边界分支的二分类 log-prob/score，因此优先使用该分支。
    if isinstance(output, (list, tuple)) and len(output) > 1:
        edge_logits = output[1]
        if edge_logits.ndim == 4 and edge_logits.size(1) > 1:
            return edge_logits.argmax(dim=1, keepdim=True) > 0
        if edge_logits.ndim == 4 and edge_logits.size(1) == 1:
            return torch.sigmoid(edge_logits) > 0.5

    # 如果模型没有边界输出，就从预测 mask 里临时提取边界。
    # 这里用 max-pool 实现近似的膨胀/腐蚀差值，得到形态学梯度。
    mask = predicted_mask.float()
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded) > 0


def binary_tensor_to_image(mask: torch.Tensor) -> np.ndarray:
    """Convert a single binary BCHW/CHW/HW tensor into uint8 grayscale."""

    # 支持 batch 张量、单样本 CHW 张量或 HW 张量，最终统一为二维灰度图。
    mask = mask.detach().cpu()
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    return (mask.numpy().astype(np.uint8) * 255)


def continuous_tensor_to_image(value: torch.Tensor, normalize: bool = False) -> np.ndarray:
    """Convert a single continuous tensor into uint8 grayscale."""

    # 距离图是连续值，不应该像 mask 一样二值化。
    # 预测距离图可能没有自然落在 [0, 1]，所以可选 min-max 拉伸增强可见性。
    value = value.detach().float().cpu()
    if value.ndim == 4:
        value = value[0, 0]
    elif value.ndim == 3:
        value = value[0]

    if normalize:
        value_min = value.min()
        value_max = value.max()
        # 常量图没有动态范围，直接显示成全黑，避免除以接近 0 的数产生噪声。
        if (value_max - value_min) > 1e-6:
            value = (value - value_min) / (value_max - value_min)
        else:
            value = torch.zeros_like(value)
    else:
        # 真实距离图在数据预处理阶段已经缩放到 [0, 1]，这里仅做保险裁剪。
        value = value.clamp(0, 1)

    return np.rint(value.numpy() * 255).astype(np.uint8)


def distance_from_output(output, predicted_mask: torch.Tensor) -> torch.Tensor:
    """Use model distance output when available, otherwise return an empty map."""

    # HBGNet 的第三个输出是距离图分支；如果加载了不带辅助输出的模型，就返回空图兜底。
    if isinstance(output, (list, tuple)) and len(output) > 2:
        distance = output[2]
        if distance.ndim == 4:
            return distance
    return torch.zeros_like(predicted_mask, dtype=torch.float32)


def make_visualization(
    rgb: np.ndarray,
    predicted_mask: np.ndarray,
    predicted_boundary: np.ndarray,
    predicted_distance: np.ndarray,
    target_mask: np.ndarray,
    target_boundary: np.ndarray,
    target_distance: np.ndarray,
) -> Image.Image:
    """Create a two-row prediction/target visualization."""

    # PIL 的拼接都使用 RGB 模式。mask/boundary/distance 本质是单通道灰度，
    # 先转 RGB 可以让它们和原图在同一张 canvas 上无缝粘贴。
    rgb_image = Image.fromarray(rgb, mode="RGB")
    predicted_mask_image = Image.fromarray(predicted_mask, mode="L").convert("RGB")
    predicted_boundary_image = Image.fromarray(predicted_boundary, mode="L").convert("RGB")
    predicted_distance_image = Image.fromarray(predicted_distance, mode="L").convert("RGB")
    target_mask_image = Image.fromarray(target_mask, mode="L").convert("RGB")
    target_boundary_image = Image.fromarray(target_boundary, mode="L").convert("RGB")
    target_distance_image = Image.fromarray(target_distance, mode="L").convert("RGB")

    width, height = rgb_image.size
    label_height = 28
    # 第一行是模型预测，第二行是真实标签；列顺序保持一致，便于上下对比。
    panels = [
        [
            ("image", rgb_image),
            ("pred_mask", predicted_mask_image),
            ("pred_boundary", predicted_boundary_image),
            ("pred_distance", predicted_distance_image),
        ],
        [
            ("image", rgb_image),
            ("target_mask", target_mask_image),
            ("target_boundary", target_boundary_image),
            ("target_distance", target_distance_image),
        ],
    ]

    # 模型输出尺寸理论上应与输入一致，但如果未来模型改了上采样策略，
    # 这里会把标签/预测图拉回原图尺寸，避免拼图时报错或尺寸错位。
    for row in panels:
        for index, (label, panel) in enumerate(row):
            if index > 0 and panel.size != rgb_image.size:
                row[index] = (label, panel.resize((width, height), Image.Resampling.NEAREST))

    # 每个 panel 顶部预留 label_height 用于写标题，主体图像仍保持原始宽高。
    row_height = height + label_height
    canvas = Image.new("RGB", (width * 4, row_height * 2), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, row in enumerate(panels):
        row_y = row_index * row_height
        for col_index, (label, panel) in enumerate(row):
            x = col_index * width
            canvas.paste(panel, (x, row_y + label_height))
            draw.text((x + 8, row_y + 8), label, fill=(0, 0, 0))

    return canvas


def resolve_device(device: str) -> torch.device:
    """Resolve a config device string to a torch.device."""

    # auto 优先使用 CUDA；没有 GPU 时自动回退到 CPU，方便同一份配置跨机器运行。
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def predict(config: dict | None = None) -> None:
    """Load a checkpoint, predict FTW samples, and save visualization PNG files."""

    config = normalize_config(config)
    device = resolve_device(config["device"])
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 直接使用训练阶段同一个 FtwDataset，这样 image/mask/boundary/dist 的读取、
    # ImageNet 归一化和多国家合并逻辑都保持一致。
    dataset = FtwDataset(
        data_root=config["data_root"],
        country=config["country"],
        split=config["split"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # MInterface.load_from_checkpoint 会恢复 LightningModule 及其内部 HBGNet。
    # return_aux_outputs=True 确保推理时可以拿到 mask、boundary、distance 三个输出。
    model = MInterface.load_from_checkpoint(
        str(config["checkpoint"]),
        map_location=device,
        return_aux_outputs=config["return_aux_outputs"],
    )
    model.to(device)
    model.eval()

    saved = 0
    with torch.inference_mode():
        for names, images, target_masks, target_boundaries, target_distances in dataloader:
            images = images.to(device)
            output = model(images)
            # output 既可能是 HBGNet 的三输出列表，也可能是普通单输出张量。
            mask_logits = output[0] if isinstance(output, (list, tuple)) else output
            predicted_masks = mask_from_logits(mask_logits, config["mask_threshold"])
            predicted_boundaries = boundary_from_output_or_mask(output, predicted_masks)
            predicted_distances = distance_from_output(output, predicted_masks)

            for item_index, name in enumerate(names):
                # 每个 batch 内逐样本保存，避免 batch_size > 1 时不同样本混到同一张图。
                rgb = tensor_to_rgb_image(images[item_index])
                predicted_mask = binary_tensor_to_image(predicted_masks[item_index])
                predicted_boundary = binary_tensor_to_image(predicted_boundaries[item_index])
                predicted_distance = continuous_tensor_to_image(
                    predicted_distances[item_index],
                    normalize=True,
                )
                target_mask = binary_tensor_to_image(target_masks[item_index])
                target_boundary = binary_tensor_to_image(target_boundaries[item_index])
                target_distance = continuous_tensor_to_image(target_distances[item_index])
                visualization = make_visualization(
                    rgb,
                    predicted_mask,
                    predicted_boundary,
                    predicted_distance,
                    target_mask,
                    target_boundary,
                    target_distance,
                )
                output_path = output_dir / f"{_safe_name(name)}.png"
                visualization.save(output_path)
                saved += 1

                # max_samples 用于快速抽样检查预测效果；None 表示保存整个 split。
                if config["max_samples"] is not None and saved >= config["max_samples"]:
                    print(f"Saved {saved} prediction visualizations to {output_dir}")
                    return

    print(f"Saved {saved} prediction visualizations to {output_dir}")


def main(config: dict | None = None) -> None:
    predict(config or PREDICT_CONFIG_DEFAULTS)


if __name__ == "__main__":
    main()
