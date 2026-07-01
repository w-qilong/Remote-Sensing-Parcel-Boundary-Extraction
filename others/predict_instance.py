"""加载实例分割 checkpoint，并交互式浏览 FBIS-22M 测试集预测结果。

本脚本的可视化风格参考 ``data/fbis22m_dataset.py`` 末尾的手动检查入口：
同一个窗口内展示原图、GT 实例、预测实例、边界概率以及距离图，并通过键盘左右键切换
测试集样本。

示例：

```bash
uv run python others/predict_instance.py \
  --checkpoint logs/field_dino_mask2former_fbis22m/version_0/checkpoints/best-epoch=00.ckpt
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# 直接执行 ``python others/predict_instance.py`` 时，Python 默认只把 others/
# 放入 sys.path；这里显式加入项目根目录，确保可以导入 data 和 model 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fbis22m_dataset import FBIS22MInstanceDataset, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model import MInterface  # noqa: E402


DEFAULT_DATA_ROOT = Path("datasets") / "FBIS-22M"
FIGURE_TITLE = "FBIS-22M checkpoint prediction browser"


def _str_to_bool(value: str | bool) -> bool:
    """把命令行中的字符串布尔值转换为 bool。"""

    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def _select_device(device_name: str) -> torch.device:
    """根据用户参数选择推理设备。"""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    dino_pretrained: bool | None = None,
) -> MInterface:
    """从 Lightning checkpoint 恢复 ``MInterface`` 并切换到 eval 模式。

    ``dino_pretrained`` 默认为 None，表示完全使用 checkpoint 保存的超参数。若本机
    无法联网或不希望初始化阶段尝试加载预训练 DINO，可传入 ``False`` 覆盖该参数。
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_kwargs: dict[str, Any] = {"map_location": device}
    if dino_pretrained is not None:
        load_kwargs["dino_pretrained"] = dino_pretrained

    model = MInterface.load_from_checkpoint(str(checkpoint_path), **load_kwargs)
    model.to(device)
    model.eval()
    return model


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """把 Dataset 输出的归一化 ``CHW`` 图像还原为 matplotlib 可显示的 RGB。"""

    image = image.detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _instance_id_map(masks: torch.Tensor) -> np.ndarray:
    """把 ``[N,H,W]`` 二值实例 mask 合成一张实例 ID 图。"""

    masks_np = masks.detach().cpu().numpy()
    if masks_np.shape[0] == 0:
        return np.zeros(masks_np.shape[1:], dtype=np.int32)

    instance_map = np.zeros(masks_np.shape[1:], dtype=np.int32)
    for instance_index, mask in enumerate(masks_np, start=1):
        # 后出现的实例覆盖前面的实例，和 Dataset 中快速可视化逻辑保持一致。
        instance_map[mask > 0] = instance_index
    return instance_map


def _predicted_instance_id_map(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    mask_threshold: float,
    min_area: int,
) -> tuple[np.ndarray, list[float]]:
    """根据 query mask 和置信度生成预测实例 ID 图。

    返回值：
    - ``instance_map``: 背景为 0，预测实例从 1 开始编号；
    - ``kept_scores``: 与实例编号顺序对应的置信度列表。
    """

    if masks.numel() == 0:
        height, width = masks.shape[-2:] if masks.ndim >= 2 else (0, 0)
        return np.zeros((height, width), dtype=np.int32), []

    order = torch.argsort(scores, descending=True)
    instance_map = np.zeros(tuple(masks.shape[-2:]), dtype=np.int32)
    kept_scores: list[float] = []

    for query_index in order.tolist():
        binary_mask = (masks[query_index].sigmoid() >= mask_threshold).detach().cpu().numpy()
        area = int(binary_mask.sum())
        if area < min_area:
            continue

        instance_id = len(kept_scores) + 1
        # 只填还没有被更高置信度实例占据的位置，减少重叠 query 对显示结果的干扰。
        visible_mask = binary_mask & (instance_map == 0)
        if int(visible_mask.sum()) < min_area:
            continue
        instance_map[visible_mask] = instance_id
        kept_scores.append(float(scores[query_index].detach().cpu()))

    return instance_map, kept_scores


def _predict_sample(
    model: MInterface,
    sample: dict[str, Any],
    device: torch.device,
    *,
    score_threshold: float,
    mask_threshold: float,
    top_k: int,
    min_area: int,
) -> dict[str, Any]:
    """对单个 Dataset 样本做前向推理并整理可视化需要的预测结果。"""

    image = sample["image"].to(device)
    height, width = image.shape[-2:]

    with torch.inference_mode():
        outputs = model(image.unsqueeze(0))

    if not isinstance(outputs, dict):
        raise ValueError("Instance prediction requires model output to be a dictionary.")

    pred_logits = outputs["pred_logits"][0]
    pred_masks = outputs["pred_masks"][0]
    boundary_logits = outputs.get("boundary_logits")
    distance_map = outputs.get("distance_map")

    class_prob = pred_logits.softmax(dim=-1)
    foreground_prob = class_prob[:, :-1]
    if foreground_prob.shape[-1] == 0:
        scores = 1.0 - class_prob[:, -1]
    else:
        scores = foreground_prob.max(dim=-1).values

    keep = scores >= score_threshold
    kept_scores = scores[keep]
    kept_masks = pred_masks[keep]
    if kept_scores.numel() > top_k:
        top_scores, top_indices = torch.topk(kept_scores, k=top_k)
        kept_scores = top_scores
        kept_masks = kept_masks[top_indices]

    # 部分 checkpoint 可能输出与输入略不同的 mask 尺寸；展示前统一回原图尺寸。
    if kept_masks.numel() > 0 and kept_masks.shape[-2:] != (height, width):
        kept_masks = F.interpolate(
            kept_masks[:, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    pred_instance_map, ordered_scores = _predicted_instance_id_map(
        kept_masks,
        kept_scores,
        mask_threshold=mask_threshold,
        min_area=min_area,
    )

    if boundary_logits is None:
        boundary_prob = None
    else:
        boundary_logit = boundary_logits[0:1]
        if boundary_logit.shape[-2:] != (height, width):
            boundary_logit = F.interpolate(
                boundary_logit,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        # boundary_head 默认输出背景/边界两个通道。用“边界类 - 背景类”得到二值边界 logit。
        if boundary_logit.shape[1] == 2:
            boundary_binary_logit = boundary_logit[:, 1:2] - boundary_logit[:, 0:1]
        else:
            boundary_binary_logit = boundary_logit
        boundary_prob = boundary_binary_logit.sigmoid()[0, 0].detach().cpu().numpy()

    if distance_map is None:
        predicted_distance = None
    else:
        distance = distance_map[0:1]
        if distance.shape[-2:] != (height, width):
            distance = F.interpolate(
                distance,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        predicted_distance = distance[0, 0].detach().cpu().numpy()

    return {
        "pred_instance_map": pred_instance_map,
        "pred_scores": ordered_scores,
        "boundary_prob": boundary_prob,
        "distance_map": predicted_distance,
        "raw_query_count": int(pred_logits.shape[0]),
        "kept_query_count": int(kept_scores.numel()),
    }


def _render_sample(
    axes: np.ndarray,
    dataset: FBIS22MInstanceDataset,
    model: MInterface,
    device: torch.device,
    index: int,
    *,
    score_threshold: float,
    mask_threshold: float,
    top_k: int,
    min_area: int,
) -> None:
    """渲染一个测试集样本的 GT 与预测结果。"""

    sample = dataset[index]
    instances = sample["instances"]
    masks = instances["masks"]  # type: ignore[index]

    image_np = _image_to_numpy(sample["image"])  # type: ignore[arg-type]
    gt_instance_map = _instance_id_map(masks)
    gt_boundary = sample["boundary"].squeeze(0).detach().cpu().numpy()  # type: ignore[union-attr]
    prediction = _predict_sample(
        model,
        sample,
        device,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        top_k=top_k,
        min_area=min_area,
    )

    for axis in axes.ravel():
        axis.clear()
        axis.axis("off")

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Image")

    axes[0, 1].imshow(gt_instance_map, cmap="tab20")
    axes[0, 1].set_title(f"GT instances: {int(masks.shape[0])}")

    axes[0, 2].imshow(image_np)
    axes[0, 2].imshow(prediction["pred_instance_map"], cmap="tab20", alpha=0.45)
    axes[0, 2].set_title(f"Pred instances: {len(prediction['pred_scores'])}")

    axes[1, 0].imshow(gt_boundary, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("GT boundary")

    boundary_prob = prediction["boundary_prob"]
    if boundary_prob is None:
        axes[1, 1].text(0.5, 0.5, "No boundary_logits", ha="center", va="center")
    else:
        axes[1, 1].imshow(boundary_prob, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Pred boundary prob")

    predicted_distance = prediction["distance_map"]
    if predicted_distance is None:
        axes[1, 2].text(0.5, 0.5, "No distance_map", ha="center", va="center")
    else:
        axes[1, 2].imshow(predicted_distance, cmap="magma")
    axes[1, 2].set_title("Pred distance")

    file_name = sample["file_name"]
    score_preview = ", ".join(f"{score:.2f}" for score in prediction["pred_scores"][:5])
    if not score_preview:
        score_preview = "none"
    axes[0, 2].text(
        0.01,
        0.99,
        f"scores: {score_preview}",
        transform=axes[0, 2].transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
    )

    axes[0, 0].figure.suptitle(
        f"{index + 1}/{len(dataset)}  {file_name}  "
        f"queries={prediction['raw_query_count']} kept={prediction['kept_query_count']}  "
        f"score>={score_threshold:.2f} mask>={mask_threshold:.2f}"
    )
    axes[0, 0].figure.tight_layout()
    axes[0, 0].figure.canvas.draw_idle()
    print(f"Showing {index + 1}/{len(dataset)} {file_name} | pred_instances={len(prediction['pred_scores'])}")


def browse_predictions(
    checkpoint_path: Path,
    *,
    data_root: Path,
    parts: list[str],
    resize_size: int | None,
    max_samples: int | None,
    start_index: int,
    device_name: str,
    score_threshold: float,
    mask_threshold: float,
    top_k: int,
    min_area: int,
    dino_pretrained: bool | None,
) -> None:
    """加载模型和测试集，并启动左右键交互式预测浏览器。"""

    device = _select_device(device_name)
    print(f"Loading checkpoint on {device}: {checkpoint_path}")
    model = _load_checkpoint_model(checkpoint_path, device, dino_pretrained=dino_pretrained)

    dataset = FBIS22MInstanceDataset(
        data_root=str(data_root),
        parts=parts,
        split="test",
        max_samples=max_samples,
        resize_size=resize_size,
        return_image_path=True,
    )
    if len(dataset) == 0:
        raise FileNotFoundError("FBIS-22M test split is empty.")

    current_index = start_index % len(dataset)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), num=FIGURE_TITLE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(FIGURE_TITLE)

    def render(index: int) -> None:
        _render_sample(
            axes,
            dataset,
            model,
            device,
            index,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            top_k=top_k,
            min_area=min_area,
        )

    def on_key_press(event) -> None:
        nonlocal current_index
        if event.key in {"left", "a"}:
            current_index = (current_index - 1) % len(dataset)
            render(current_index)
        elif event.key in {"right", "d", " "}:
            current_index = (current_index + 1) % len(dataset)
            render(current_index)
        elif event.key in {"escape", "q"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    render(current_index)
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    """构建预测可视化脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Browse FBIS-22M checkpoint predictions with arrow keys.")
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/field_dino_mask2former_fbis22m/version_0/checkpoints/best-02.ckpt"), help="Lightning checkpoint path.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to datasets/FBIS-22M.")
    parser.add_argument("--parts", nargs="+", default=["all"], help="FBIS-22M parts to include, or all.")
    parser.add_argument("--resize-size", type=int, default=512, help="Resize size used by the dataset. Use 0 to disable.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional number of test samples to browse.")
    parser.add_argument("--start-index", type=int, default=0, help="Initial sample index.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Foreground query confidence threshold.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Sigmoid mask threshold.")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum number of kept queries before area filtering.")
    parser.add_argument("--min-area", type=int, default=16, help="Minimum visible mask area in pixels.")
    parser.add_argument(
        "--dino-pretrained",
        nargs="?",
        const=True,
        default=None,
        type=_str_to_bool,
        help="Override checkpoint hparam dino_pretrained. Use false to avoid pretrained init downloads.",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    resize_size = None if args.resize_size == 0 else args.resize_size
    browse_predictions(
        args.checkpoint,
        data_root=args.data_root,
        parts=args.parts,
        resize_size=resize_size,
        max_samples=args.max_samples,
        start_index=args.start_index,
        device_name=args.device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        top_k=args.top_k,
        min_area=args.min_area,
        dino_pretrained=args.dino_pretrained,
    )


if __name__ == "__main__":
    main()
