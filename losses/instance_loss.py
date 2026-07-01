"""Field-DINO-Mask2Former 实例分割损失。

本文件实现与 ``model/instance_model.py`` 输出结构对应的训练损失。整体思路遵循
设计文档中的 DETR/Mask2Former 风格：

1. 对每张图的 ``Q`` 个预测 query 和 ``M`` 个真实田块实例做 Hungarian matching；
2. 对匹配成功的 query 计算类别、实例 mask、Dice、box L1 和 GIoU 损失；
3. 对未匹配 query 监督为 no-object；
4. 额外监督边界图和距离图；
5. 可选对 decoder 中间层 ``aux_outputs`` 重复实例损失，增强深层 decoder 训练稳定性。

为了避免高分辨率 mask 在 Hungarian matching 中构造巨大的
``[num_queries, num_targets, H * W]`` 张量，本实现默认采用 point-based mask
loss：每张图随机采样固定数量的点，只在这些点上计算 mask BCE/Dice 代价和损失。

默认 ``forward`` 返回总损失 Tensor，便于直接接入 Lightning；如果构造时设置
``return_dict=True``，则返回包含总损失和各分项的字典，方便调试权重比例。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - 只有实际调用 Hungarian matching 时才需要报错。
    linear_sum_assignment = None


TargetList = Sequence[Mapping[str, torch.Tensor]]


def _safe_num_masks(targets: TargetList) -> float:
    """统计当前 batch 中真实实例数量，并至少返回 1。

    损失按真实实例数归一化可以避免“实例很多的图片”主导梯度；空标注 batch 时使用 1
    作为分母，确保 no-object 分类损失仍然可计算。
    """

    num_masks = sum(int(target["masks"].shape[0]) for target in targets)
    return float(max(num_masks, 1))


def _as_tensor_target_list(targets: Any) -> list[Mapping[str, torch.Tensor]]:
    """从完整 batch 或实例列表中取出 matcher 需要的 targets。

    支持两种输入：
    - ``batch`` 字典：包含 ``instances``、``boundary``、``distance``；
    - ``list[instances]``：每个元素包含 ``masks``、``boxes``、``labels``。
    """

    if isinstance(targets, Mapping) and "instances" in targets:
        instances = targets["instances"]
    else:
        instances = targets

    if not isinstance(instances, Sequence):
        raise TypeError("InstanceLoss targets must be a batch dict or a list of instance dictionaries.")
    return list(instances)


def _stack_or_pad_dense_target(
    values: Any,
    output_size: tuple[int, int],
    device: torch.device,
    interpolation_mode: str = "nearest",
) -> torch.Tensor | None:
    """把 dense target 统一成 ``[B, C, H, W]``。

    ``Fbis22mDataset.collate_fn`` 在样本同尺寸时会直接 stack；不同尺寸时会保持 list。
    这里对 list 情况做右下 padding，并在最后插值到模型输出尺寸，保证辅助损失可用。
    """

    if values is None:
        return None

    if torch.is_tensor(values):
        target = values.to(device)
    else:
        tensors = [value.to(device) for value in values]
        channels = tensors[0].shape[0]
        max_height = max(tensor.shape[-2] for tensor in tensors)
        max_width = max(tensor.shape[-1] for tensor in tensors)
        target = tensors[0].new_zeros((len(tensors), channels, max_height, max_width))
        for index, tensor in enumerate(tensors):
            height, width = tensor.shape[-2:]
            target[index, :, :height, :width] = tensor

    if target.shape[-2:] != output_size:
        # 语义/边界是离散标签，默认 nearest；距离图是连续监督，调用方会传入 bilinear。
        if interpolation_mode == "bilinear":
            target = F.interpolate(target.float(), size=output_size, mode=interpolation_mode, align_corners=False)
        else:
            target = F.interpolate(target.float(), size=output_size, mode=interpolation_mode)
    return target


def _sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    """二值 mask 的 BCEWithLogits 损失。

    输入 ``inputs`` 是 logits，不需要提前 sigmoid。先对每个实例的空间维求平均，再
    对实例维求和并除以真实实例数，和 DETR/Mask2Former 的归一化习惯一致。
    """

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.flatten(1).mean(1).sum() / num_masks


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    """二值 mask 的 Dice 损失，使用 logits 输入。"""

    probabilities = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def _sample_point_coords(num_points: int, device: torch.device) -> torch.Tensor:
    """随机生成归一化点坐标，形状为 ``[num_points, 2]``。

    坐标顺序是 ``x, y``，取值范围为 ``[0, 1]``。后续采样时会转换成
    ``grid_sample`` 需要的 ``[-1, 1]`` 坐标。
    """

    return torch.rand(num_points, 2, device=device)


def _point_sample_masks(masks: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """在若干归一化点上采样 mask/logit。

    参数：
    - ``masks``: ``[N, H, W]``，可以是预测 logits，也可以是 GT 二值 mask；
    - ``point_coords``: ``[P, 2]``，归一化 ``x, y`` 坐标。

    返回：
    - ``[N, P]``，每个 mask 在 P 个点上的采样值。

    这里使用双线性采样，使采样点不必落在整数像素中心；这和 Mask2Former/PointRend
    的 point-based mask loss 思路一致。
    """

    if masks.numel() == 0:
        return masks.new_zeros((masks.shape[0], point_coords.shape[0]))

    grid = point_coords.mul(2).sub(1).view(1, -1, 1, 2)
    grid = grid.expand(masks.shape[0], -1, -1, -1)
    sampled = F.grid_sample(
        masks[:, None].float(),
        grid,
        mode="bilinear",
        align_corners=False,
    )
    return sampled[:, 0, :, 0]


def _pairwise_sigmoid_ce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算采样点上的 pairwise BCEWithLogits 代价。

    ``inputs`` 形状为 ``[Q, P]``，``targets`` 形状为 ``[M, P]``。直接展开成
    ``[Q, M, P]`` 会占用大量显存，因此这里使用 BCE 的等价形式：

    ``BCE(logit, y) = softplus(logit) - logit * y``

    这样只需要一次矩阵乘法就能得到 ``[Q, M]`` 代价矩阵。
    """

    num_points = inputs.shape[1]
    positive_term = F.softplus(inputs).mean(dim=1)[:, None]
    target_term = torch.einsum("qp,mp->qm", inputs, targets) / max(num_points, 1)
    return positive_term - target_term


def _pairwise_dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算采样点上的 pairwise Dice 代价，返回 ``[Q, M]``。"""

    probabilities = inputs.sigmoid()
    numerator = 2 * torch.einsum("qp,mp->qm", probabilities, targets)
    denominator = probabilities.sum(dim=1)[:, None] + targets.sum(dim=1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def _sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """边界辅助分支使用的二值 Focal loss。

    边界像素通常远少于非边界像素，Focal loss 能降低大量易分类背景像素的权重，
    让模型更关注边界附近的困难样本。
    """

    probabilities = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """把归一化 ``cx, cy, w, h`` 转成 ``x0, y0, x1, y1``。"""

    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ),
        dim=-1,
    )


def _box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """把 ``x0, y0, x1, y1`` 转成 ``cx, cy, w, h``。"""

    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(
        (
            (x0 + x1) * 0.5,
            (y0 + y1) * 0.5,
            (x1 - x0).clamp(min=0),
            (y1 - y0).clamp(min=0),
        ),
        dim=-1,
    )


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    """计算 ``xyxy`` box 面积。"""

    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组 ``xyxy`` box 的 generalized IoU 矩阵。

    返回形状为 ``[N, M]``，第 ``i,j`` 个元素表示第 i 个预测框和第 j 个 GT 框的 GIoU。
    """

    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    enclosing_left_top = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_right_bottom = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_wh = (enclosing_right_bottom - enclosing_left_top).clamp(min=0)
    enclosing_area = enclosing_wh[:, :, 0] * enclosing_wh[:, :, 1]

    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)


def _normalize_target_boxes(target: Mapping[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """把数据集中的 GT box 统一成归一化 ``cxcywh`` 和 ``xyxy``。

    当前 FBIS-22M Dataset 生成的是像素坐标 ``xyxy``。为了和模型输出的归一化
    ``pred_boxes`` 对齐，这里根据 target masks 的 H/W 自动归一化。若外部数据集
    已经提供 [0, 1] 范围内的 box，则保持其数值尺度不变。
    """

    boxes = target["boxes"].to(device).float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4), boxes.reshape(0, 4)

    masks = target["masks"]
    height, width = masks.shape[-2:]
    boxes_xyxy = boxes.clone()
    if boxes_xyxy.max() > 1.5:
        scale = boxes_xyxy.new_tensor([width, height, width, height]).clamp(min=1)
        boxes_xyxy = boxes_xyxy / scale

    boxes_xyxy = boxes_xyxy.clamp(0, 1)
    boxes_cxcywh = _box_xyxy_to_cxcywh(boxes_xyxy)
    return boxes_cxcywh, boxes_xyxy


class HungarianMatcher(nn.Module):
    """根据类别、mask、Dice、box 和 GIoU 代价完成二分图匹配。

    matcher 只决定“哪个 query 对应哪个真实实例”，本身不参与反向传播，因此整段计算
    在 ``torch.no_grad`` 中执行。输出是每张图一组索引：
    ``(pred_indices, target_indices)``。
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        cost_box: float = 2.0,
        cost_giou: float = 2.0,
        num_mask_points: int = 4096,
    ) -> None:
        super().__init__()
        if num_mask_points <= 0:
            raise ValueError("num_mask_points must be a positive integer.")
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        self.num_mask_points = num_mask_points

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: TargetList,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if linear_sum_assignment is None:
            raise ImportError("InstanceLoss requires scipy for Hungarian matching. Please install scipy.")

        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        pred_boxes = outputs["pred_boxes"]
        batch_size, num_queries = pred_logits.shape[:2]

        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index in range(batch_size):
            target = targets[batch_index]
            target_masks = target["masks"].to(pred_masks.device).float()
            num_targets = int(target_masks.shape[0])
            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_logits.device)
                indices.append((empty, empty))
                continue

            # GT label 只允许落在真实类别范围内，最后一类 no-object 只能分配给未匹配 query。
            max_valid_class = pred_logits.shape[-1] - 2
            target_labels = target["labels"].to(pred_logits.device).long().clamp(min=0, max=max_valid_class)
            target_boxes_cxcywh, target_boxes_xyxy = _normalize_target_boxes(target, pred_logits.device)

            # 类别代价：取预测属于 GT 类别的概率负值，概率越大代价越小。
            class_prob = pred_logits[batch_index].softmax(dim=-1)
            cost_class = -class_prob[:, target_labels]

            # mask 代价只在采样点上计算，避免构造 [Q, M, H*W] 的巨大张量。
            # 点坐标是归一化坐标，因此预测 mask 和 GT mask 即使分辨率不同也能各自采样。
            query_masks = pred_masks[batch_index]
            point_coords = _sample_point_coords(self.num_mask_points, pred_masks.device)
            out_masks = _point_sample_masks(query_masks, point_coords)
            tgt_masks = _point_sample_masks(target_masks, point_coords)

            # BCE/Dice 代价：逐 query/target 成对计算，得到 [Q, M]，但不展开到 [Q, M, P]。
            cost_mask = _pairwise_sigmoid_ce_cost(out_masks, tgt_masks)
            cost_dice = _pairwise_dice_cost(out_masks, tgt_masks)

            # box/GIoU 代价在归一化坐标上计算，和输入图像实际大小解耦。
            cost_box = torch.cdist(pred_boxes[batch_index], target_boxes_cxcywh, p=1)
            pred_boxes_xyxy = _box_cxcywh_to_xyxy(pred_boxes[batch_index]).clamp(0, 1)
            cost_giou = -_generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)

            total_cost = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
                + self.cost_box * cost_box
                + self.cost_giou * cost_giou
            )
            pred_ind, target_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(pred_ind, dtype=torch.long, device=pred_logits.device),
                    torch.as_tensor(target_ind, dtype=torch.long, device=pred_logits.device),
                )
            )
        return indices


class InstanceLoss(nn.Module):
    """Field-DINO-Mask2Former 的总损失函数。

    主要输入：
    - ``outputs``: ``InstanceModel.forward`` 返回的字典；
    - ``targets``: ``Fbis22mDataset.collate_fn`` 返回的 batch 字典，或 ``list[instances]``。

    默认权重与设计文档保持一致：
    - class=2, mask=5, dice=5, box=2, giou=2；
    - boundary=1.0, distance=0.2；
    - aux decoder layers 使用和主实例分支相同的权重。
    """

    def __init__(
        self,
        num_classes: int = 1,
        eos_coef: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        box_weight: float = 2.0,
        giou_weight: float = 2.0,
        boundary_weight: float = 1.0,
        distance_weight: float = 0.2,
        aux_weight: float = 1.0,
        boundary_focal_alpha: float = 0.25,
        boundary_focal_gamma: float = 2.0,
        num_mask_points: int = 4096,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        if num_mask_points <= 0:
            raise ValueError("num_mask_points must be a positive integer.")
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.box_weight = box_weight
        self.giou_weight = giou_weight
        self.boundary_weight = boundary_weight
        self.distance_weight = distance_weight
        self.aux_weight = aux_weight
        self.boundary_focal_alpha = boundary_focal_alpha
        self.boundary_focal_gamma = boundary_focal_gamma
        self.num_mask_points = num_mask_points
        self.return_dict = return_dict

        self.matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_box=box_weight,
            cost_giou=giou_weight,
            num_mask_points=num_mask_points,
        )

        # 类别 CE 中最后一类是 no-object。eos_coef 越小，未匹配 query 的权重越低。
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _loss_labels(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: TargetList,
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """计算 query 分类损失，未匹配 query 统一监督为 no-object。"""

        pred_logits = outputs["pred_logits"]
        batch_size, num_queries = pred_logits.shape[:2]
        no_object_class = pred_logits.shape[-1] - 1
        target_classes = torch.full(
            (batch_size, num_queries),
            no_object_class,
            dtype=torch.long,
            device=pred_logits.device,
        )

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            labels = targets[batch_index]["labels"].to(pred_logits.device).long()
            target_classes[batch_index, src_idx] = labels[tgt_idx].clamp(min=0, max=self.num_classes - 1)

        return F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(pred_logits.device),
        )

    def _loss_masks(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: TargetList,
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
        num_masks: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算匹配实例的 mask BCE 和 Dice 损失。"""

        pred_masks = outputs["pred_masks"]
        src_masks: list[torch.Tensor] = []
        tgt_masks: list[torch.Tensor] = []
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_masks = targets[batch_index]["masks"].to(pred_masks.device).float()
            selected_pred = pred_masks[batch_index, src_idx]
            selected_tgt = target_masks[tgt_idx]
            # 训练 mask loss 也使用采样点，避免匹配后仍把所有实例 mask 全图拼接到显存中。
            point_coords = _sample_point_coords(self.num_mask_points, pred_masks.device)
            src_masks.append(_point_sample_masks(selected_pred, point_coords))
            tgt_masks.append(_point_sample_masks(selected_tgt, point_coords))

        if not src_masks:
            zero = pred_masks.sum() * 0.0
            return zero, zero

        src = torch.cat(src_masks, dim=0)
        tgt = torch.cat(tgt_masks, dim=0)
        return _sigmoid_ce_loss(src, tgt, num_masks), _dice_loss(src, tgt, num_masks)

    def _loss_boxes(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: TargetList,
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
        num_masks: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算匹配实例的 box L1 和 GIoU 损失。"""

        pred_boxes = outputs["pred_boxes"]
        src_boxes: list[torch.Tensor] = []
        tgt_boxes_cxcywh: list[torch.Tensor] = []
        tgt_boxes_xyxy: list[torch.Tensor] = []

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_cxcywh, target_xyxy = _normalize_target_boxes(targets[batch_index], pred_boxes.device)
            src_boxes.append(pred_boxes[batch_index, src_idx])
            tgt_boxes_cxcywh.append(target_cxcywh[tgt_idx])
            tgt_boxes_xyxy.append(target_xyxy[tgt_idx])

        if not src_boxes:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        src = torch.cat(src_boxes, dim=0)
        tgt_cxcywh = torch.cat(tgt_boxes_cxcywh, dim=0)
        tgt_xyxy = torch.cat(tgt_boxes_xyxy, dim=0)

        loss_box = F.l1_loss(src, tgt_cxcywh, reduction="none").sum() / num_masks
        src_xyxy = _box_cxcywh_to_xyxy(src).clamp(0, 1)
        giou = _generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1 - torch.diag(giou)).sum() / num_masks
        return loss_box, loss_giou

    def _instance_losses(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: TargetList,
        prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        """执行一次 Hungarian matching，并返回实例分支的所有损失项。"""

        indices = self.matcher(outputs, targets)
        num_masks = _safe_num_masks(targets)

        loss_ce = self._loss_labels(outputs, targets, indices)
        loss_mask, loss_dice = self._loss_masks(outputs, targets, indices, num_masks)
        loss_box, loss_giou = self._loss_boxes(outputs, targets, indices, num_masks)

        return {
            f"{prefix}loss_ce": loss_ce,
            f"{prefix}loss_mask": loss_mask,
            f"{prefix}loss_dice": loss_dice,
            f"{prefix}loss_box": loss_box,
            f"{prefix}loss_giou": loss_giou,
        }

    def _auxiliary_dense_losses(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch_targets: Any,
    ) -> dict[str, torch.Tensor]:
        """计算 boundary 和 distance 两个 dense auxiliary 损失。"""

        device = outputs["pred_logits"].device
        output_size = outputs["pred_masks"].shape[-2:]
        losses: dict[str, torch.Tensor] = {}

        if isinstance(batch_targets, Mapping):
            boundary_target = _stack_or_pad_dense_target(batch_targets.get("boundary"), output_size, device)
            distance_target = _stack_or_pad_dense_target(
                batch_targets.get("distance"),
                output_size,
                device,
                interpolation_mode="bilinear",
            )
        else:
            boundary_target = distance_target = None

        if boundary_target is not None and "boundary_logits" in outputs:
            boundary_logits = outputs["boundary_logits"]
            if boundary_logits.shape[-2:] != boundary_target.shape[-2:]:
                boundary_logits = F.interpolate(
                    boundary_logits,
                    size=boundary_target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            # boundary_logits 有两个通道时，使用“边界类 logit - 背景类 logit”作为二值边界 logit。
            if boundary_logits.shape[1] == 2:
                boundary_binary_logits = boundary_logits[:, 1:2] - boundary_logits[:, 0:1]
            else:
                boundary_binary_logits = boundary_logits
            boundary_float = boundary_target.float()
            losses["loss_boundary"] = _sigmoid_focal_loss(
                boundary_binary_logits,
                boundary_float,
                alpha=self.boundary_focal_alpha,
                gamma=self.boundary_focal_gamma,
            ) + _dice_loss(boundary_binary_logits, boundary_float, boundary_float.shape[0])

        if distance_target is not None and "distance_map" in outputs:
            distance_map = outputs["distance_map"]
            if distance_map.shape[-2:] != distance_target.shape[-2:]:
                distance_map = F.interpolate(
                    distance_map,
                    size=distance_target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            losses["loss_distance"] = F.smooth_l1_loss(distance_map, distance_target.float())

        return losses

    def _weighted_total(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """按设计文档权重把所有分项合成总损失。"""

        first_loss = next(iter(losses.values()))
        total = first_loss.new_zeros(())
        for name, value in losses.items():
            if name.endswith("loss_ce"):
                total = total + self.class_weight * value
            elif name.endswith("loss_mask"):
                total = total + self.mask_weight * value
            elif name.endswith("loss_dice"):
                total = total + self.dice_weight * value
            elif name.endswith("loss_box"):
                total = total + self.box_weight * value
            elif name.endswith("loss_giou"):
                total = total + self.giou_weight * value
            elif name == "loss_boundary":
                total = total + self.boundary_weight * value
            elif name == "loss_distance":
                total = total + self.distance_weight * value
        return total

    def forward(self, outputs: Mapping[str, Any], targets: Any) -> torch.Tensor | dict[str, torch.Tensor]:
        """计算模型输出和 batch 标注之间的总损失。

        ``outputs`` 必须至少包含 ``pred_logits``、``pred_masks`` 和 ``pred_boxes``。
        若包含 ``aux_outputs``，会自动对每个中间 decoder 层追加辅助实例损失。
        """

        instance_targets = _as_tensor_target_list(targets)

        losses = self._instance_losses(outputs, instance_targets)
        losses.update(self._auxiliary_dense_losses(outputs, targets))

        for aux_index, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
            aux_losses = self._instance_losses(aux_outputs, instance_targets, prefix=f"aux_{aux_index}_")
            # aux loss 的权重可整体调节；分项名称保留，方便 return_dict 时定位问题层。
            for name, value in aux_losses.items():
                losses[name] = value * self.aux_weight

        total = self._weighted_total(losses)
        if self.return_dict:
            result = {"loss": total}
            result.update(losses)
            return result
        return total
