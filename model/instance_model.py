"""DINOv3 + DPT pixel decoder + Mask2Former-style instance segmentation.

本模块实现设计文档中的 ``Field-DINO-Mask2Former`` 主体结构，并遵循项目
动态加载约定：文件名 ``instance_model.py`` 对应类名 ``InstanceModel``。

模型输出是一个字典，适合后续接入 DETR/Mask2Former 风格 Hungarian matching
损失：

```text
{
    "pred_logits": Tensor[B, Q, num_classes + 1],
    "pred_masks":  Tensor[B, Q, H, W],
    "pred_boxes":  Tensor[B, Q, 4],
    "aux_outputs": list[dict],
    "semantic_logits": Tensor[B, 1, H, W],
    "boundary_logits": Tensor[B, 2, H, W],
    "distance_map": Tensor[B, 1, H, W],
}
```

其中 ``num_classes + 1`` 的最后一类是 no-object，``pred_boxes`` 使用归一化
``cx, cy, w, h`` 格式。当前文件只实现模型结构；实例匹配和损失函数应放在
``losses/`` 中单独实现。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover - 导入阶段的依赖检查，便于给出明确错误。
    raise ImportError("InstanceModel requires timm. Install project dependencies first.") from exc

from .dino_dpt import (
    ConvNormAct,
    ResidualConvUnit,
    _default_out_indices,
    _init_decoder_weights,
    _make_reassemble_layer,
)


def _init_transformer_weights(module: nn.Module) -> None:
    """初始化模型中新增加的 Transformer/query 相关层。

    这里的初始化只用于本文件中新建的 decoder、query embedding 和 prediction head。
    DINO backbone 可能已经加载了大规模预训练权重，因此不要把这个函数作用到
    ``self.backbone`` 上，否则会破坏预训练特征。
    """

    if isinstance(module, nn.Linear):
        # Linear 层采用 Xavier 初始化，是 Transformer 中较常用且稳定的默认选择。
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # query embedding/level embedding 使用小方差正态分布，避免初始 query 过大。
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm 初始化为恒等归一化：缩放系数为 1，偏置为 0。
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _as_4tuple(indices: Sequence[int] | None, model_name: str) -> tuple[int, int, int, int]:
    """把 DINO 输出层下标规范化为长度为 4 的 tuple。

    ``DptPyramidPixelDecoder`` 约定接收 4 个 ViT 中间层特征：浅层、中浅层、
    中深层、深层。若用户没有显式传入 ``dino_out_indices``，则根据模型名称使用
    ``dino_dpt.py`` 中的默认配置。
    """

    out_indices = tuple(indices) if indices is not None else _default_out_indices(model_name)
    if len(out_indices) != 4:
        raise ValueError("InstanceModel expects exactly 4 DINO feature levels.")
    return out_indices  # type: ignore[return-value]


def _pad_tensor_list(images: Sequence[torch.Tensor]) -> torch.Tensor:
    """把若干张 ``CHW`` 图像 padding 成一个 ``BCHW`` batch。

    ``Fbis22mDataset.collate_fn`` 为了支持不同尺寸影像，会把 batch 保持为 list。
    这个辅助函数在模型入口处把它们补零到同一高度和宽度，使 backbone 可以一次
    前向传播。训练时若追求效率，仍建议在 Dataset 中使用 ``resize_size``，因为
    padding 出来的背景区域也会参与计算。
    """

    if not images:
        raise ValueError("InstanceModel received an empty image list.")

    # 以 batch 中最大 H/W 作为画布尺寸；所有图像左上角对齐，剩余区域填 0。
    channels = images[0].shape[0]
    max_height = max(image.shape[-2] for image in images)
    max_width = max(image.shape[-1] for image in images)
    batch = images[0].new_zeros((len(images), channels, max_height, max_width))
    for index, image in enumerate(images):
        if image.shape[0] != channels:
            raise ValueError("All images in a batch must have the same channel count.")
        height, width = image.shape[-2:]
        # 只拷贝原始有效区域；padding 区域保持 0。
        batch[index, :, :height, :width] = image
    return batch


class MLP(nn.Module):
    """用于 mask embedding 和 box 预测的小型前馈网络。

    Mask2Former 的 query decoder 输出的是每个 query 的隐藏向量。该隐藏向量需要
    分别映射为：
    - 与像素特征做点积的 mask embedding；
    - 归一化的框坐标 ``cx, cy, w, h``。
    这两个映射都可以用结构简单的多层感知机完成。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("MLP needs at least one layer.")

        layers: list[nn.Module] = []
        for layer_index in range(num_layers):
            # 首层接收 input_dim，中间层保持 hidden_dim，末层输出任务需要的维度。
            in_dim = input_dim if layer_index == 0 else hidden_dim
            out_dim = output_dim if layer_index == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_index < num_layers - 1:
                # 末层不加激活，便于调用方自行决定是否 sigmoid/softmax。
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.apply(_init_transformer_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinePositionEmbedding2D(nn.Module):
    """为二维 feature map 构造正弦/余弦位置编码。

    DINO/ViT 特征本身已经包含位置相关信息，但当 token 特征被重新组织成像素金字塔
    后，query decoder 的 cross-attention 仍然需要明确知道每个 memory token 的
    空间位置。这里输出形状为 ``[B, hidden_dim, H, W]``，后续会 flatten 成
    ``[B, H*W, hidden_dim]`` 供 MultiheadAttention 使用。
    """

    def __init__(self, hidden_dim: int = 256, temperature: int = 10000) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("SinePositionEmbedding2D requires hidden_dim to be divisible by 4.")
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = feature.shape
        device = feature.device
        dtype = feature.dtype

        # 使用 0 到 1 的归一化坐标，避免绝对像素尺寸变化导致位置尺度不一致。
        y_embed = torch.linspace(0, 1, height, device=device, dtype=dtype).view(1, height, 1)
        x_embed = torch.linspace(0, 1, width, device=device, dtype=dtype).view(1, 1, width)
        y_embed = y_embed.expand(batch_size, height, width)
        x_embed = x_embed.expand(batch_size, height, width)

        # hidden_dim 的一半给 y，一半给 x；每个方向再拆成 sin/cos 两组频率。
        dim_t = torch.arange(self.hidden_dim // 4, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / (self.hidden_dim // 2))

        # 生成多频率位置编码：[B, H, W, C/2]，最后转回通道优先格式。
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        position = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return position[:, : self.hidden_dim]


class DptPyramidPixelDecoder(nn.Module):
    """把 4 个同 stride 的 ViT 特征转换为细化后的像素金字塔。

    DINO/ViT 的多层输出通常共享同一个 patch 网格分辨率，例如都是 stride 16。
    实例分割则更依赖高分辨率边界与低分辨率语义的结合，所以这里借鉴 DPT/FPN：
    先通过 ``reassemble`` 制造伪多尺度特征，再自顶向下融合。

    输出：
    - ``mask_features``: 高分辨率像素嵌入，默认 stride 4，用于生成实例 mask。
    - ``multi_scale_features``: 从深到浅的多尺度 memory，供 query decoder 交替注意。
    - ``pyramid``: 从浅到深的完整金字塔，供调试或后续扩展。
    """

    def __init__(self, in_channels: Sequence[int], hidden_dim: int) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("DptPyramidPixelDecoder expects exactly 4 input feature levels.")

        # 不同 DINO 规模的输出通道可能不同，先统一投影到 hidden_dim。
        self.projections = nn.ModuleList(
            [ConvNormAct(channels, hidden_dim, kernel_size=1) for channels in in_channels]
        )
        # 将同尺度 ViT 特征重组为大致对应 stride 4/8/16/32 的伪金字塔。
        self.reassemble = nn.ModuleList(
            [_make_reassemble_layer(index, hidden_dim) for index in range(len(in_channels))]
        )
        # lateral_blocks 处理每个 level 自身特征，output_blocks 处理与 top-down 融合后的结果。
        self.lateral_blocks = nn.ModuleList([ResidualConvUnit(hidden_dim) for _ in in_channels])
        self.output_blocks = nn.ModuleList([ResidualConvUnit(hidden_dim) for _ in in_channels])
        # mask_projection 生成最终用于点积成 mask 的像素嵌入。
        self.mask_projection = nn.Sequential(
            ResidualConvUnit(hidden_dim),
            ConvNormAct(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.apply(_init_decoder_weights)

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")

        # 对每层 DINO 特征执行：通道投影 -> 空间尺度重组。
        # 输出顺序保持从浅到深，典型空间尺寸约为 1/4、1/8、1/16、1/32。
        pyramid = [
            reassemble(project(feature))
            for feature, project, reassemble in zip(features, self.projections, self.reassemble)
        ]

        # FPN 式自顶向下细化：从最深层语义特征开始，逐级上采样并加到更高分辨率层。
        refined: list[torch.Tensor | None] = [None] * len(pyramid)
        top_down: torch.Tensor | None = None
        for level_index in reversed(range(len(pyramid))):
            lateral = self.lateral_blocks[level_index](pyramid[level_index])
            if top_down is not None:
                top_down = F.interpolate(
                    top_down,
                    size=lateral.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                lateral = lateral + top_down
            refined[level_index] = self.output_blocks[level_index](lateral)
            top_down = refined[level_index]

        # refined_pyramid 顺序仍为浅 -> 深；第 0 层分辨率最高，最适合生成细粒度 mask。
        refined_pyramid = [feature for feature in refined if feature is not None]
        mask_features = self.mask_projection(refined_pyramid[0])

        return {
            "mask_features": mask_features,
            # query decoder 使用深 -> 浅的 memory，并在各层 decoder 中循环访问。
            "multi_scale_features": [refined_pyramid[3], refined_pyramid[2], refined_pyramid[1]],
            "pyramid": refined_pyramid,
        }


class Mask2FormerDecoderLayer(nn.Module):
    """单层 Mask2Former 风格 query decoder。

    每个 query 可以理解为一个“候选田块实例”的可学习表示。本层更新 query 的顺序为：
    1. masked cross-attention：只关注上一轮 mask 预测认为可能属于该实例的区域；
    2. query self-attention：让不同实例 query 之间交换信息，减少重复预测；
    3. FFN：对每个 query 独立做非线性变换。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

        self.apply(_init_transformer_weights)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # cross-attention：query 带上 query_pos，memory 带上空间位置编码 memory_pos。
        cross_query = queries + query_pos
        cross_key = memory + memory_pos
        cross_out, _ = self.cross_attn(
            query=cross_query,
            key=cross_key,
            value=memory,
            attn_mask=attention_mask,
            need_weights=False,
        )
        queries = self.norm1(queries + self.dropout1(cross_out))

        # self-attention：同一张图内的 Q 个实例 query 互相通信。
        self_query = queries + query_pos
        self_out, _ = self.self_attn(
            query=self_query,
            key=self_query,
            value=queries,
            need_weights=False,
        )
        queries = self.norm2(queries + self.dropout2(self_out))

        # FFN：标准 Transformer 子层，残差连接 + LayerNorm 保持训练稳定。
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout3(ffn_out))
        return queries


class Mask2FormerQueryDecoder(nn.Module):
    """把可学习的田块 query 解码为类别、实例 mask 和框。

    输入来自像素解码器：
    - ``mask_features``: 高分辨率像素嵌入，用于和每个 query 的 mask embedding 点积；
    - ``multi_scale_features``: 3 个不同尺度的 memory，用于 decoder 层轮流 cross-attention。

    输出遵循 DETR/Mask2Former 习惯：
    - ``pred_logits``: 每个 query 的类别分数，最后一类是 no-object；
    - ``pred_masks``: 每个 query 对应的一张 mask logit；
    - ``pred_boxes``: 归一化 ``cx, cy, w, h``，辅助匹配和后处理；
    - ``aux_outputs``: 中间 decoder 层预测，训练时可做辅助监督。
    """

    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 9,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        mask_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("Mask2FormerQueryDecoder needs at least one decoder layer.")

        self.num_queries = num_queries
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mask_embed_dim = mask_embed_dim or hidden_dim

        # query_feat 是 query 的内容向量；query_pos 是 query 的位置/身份编码。
        # 二者都是可学习参数，数量 num_queries 决定模型一次最多提出多少个实例候选。
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)
        # 三个 memory level 共享 decoder，但用 level_embed 告诉模型当前来自哪个尺度。
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.position_embedding = SinePositionEmbedding2D(hidden_dim)

        # 输入 memory 通道理论上已是 hidden_dim，这里保留 1x1 投影，便于后续调整或对齐分布。
        self.input_projections = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(3)]
        )
        self.layers = nn.ModuleList(
            [
                Mask2FormerDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # no-object 类放在最后一维，和 DETR/Mask2Former 约定一致。
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed_head = MLP(hidden_dim, hidden_dim, self.mask_embed_dim, num_layers=3)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.input_projections.apply(_init_decoder_weights)
        self.apply(_init_transformer_weights)

    @staticmethod
    def _flatten_memory(feature: torch.Tensor, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把 ``[B, C, H, W]`` 的 memory 展平为注意力需要的 ``[B, H*W, C]``。"""

        memory = feature.flatten(2).transpose(1, 2)
        memory_pos = position.flatten(2).transpose(1, 2)
        return memory, memory_pos

    def _predict_heads(
        self,
        queries: torch.Tensor,
        mask_features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """根据当前 query 状态预测类别、mask 和 box。

        该函数会在 decoder 开始前和每一层更新后调用。开始前的预测用于生成第一层
        masked cross-attention 的 attention mask；每层之后的预测则用于辅助监督和
        下一层 attention mask。
        """

        # 类别头输出 [B, Q, num_classes + 1]，最后一维的末类别表示 no-object。
        class_logits = self.class_head(queries)
        # mask_embed: [B, Q, C]，表示每个 query 需要在像素特征空间中寻找的方向。
        mask_embed = self.mask_embed_head(queries)
        # 与 mask_features 做逐通道点积得到 [B, Q, H_mask, W_mask] 的 mask logits。
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        # 输出 mask 统一插值回原图尺寸，便于直接和实例标注计算损失。
        mask_logits = F.interpolate(
            mask_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        # box 采用 sigmoid 约束到 [0, 1]，格式为归一化 cxcywh。
        # 它不是生成 mask 的必要条件，但有利于 Hungarian matching 和后处理。
        box_pred = self.box_head(queries).sigmoid()
        return {
            "pred_logits": class_logits,
            "pred_masks": mask_logits,
            "pred_boxes": box_pred,
        }

    def _build_attention_mask(
        self,
        mask_logits: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """把上一轮 mask 预测转换为 ``MultiheadAttention`` 需要的布尔 attention mask。

        PyTorch 约定 ``True`` 表示禁止注意该位置。这里将 mask 概率小于 0.5 的区域
        设为不可见，使下一轮 cross-attention 更聚焦于当前 query 负责的实例区域。
        返回形状为 ``[B * num_heads, Q, H_level * W_level]``。
        """

        # 先把原图尺度 mask 缩放到当前 memory level 的空间尺寸。
        resized_masks = F.interpolate(
            mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        # PyTorch MultiheadAttention 中 True 表示“不允许 attend”。
        attention_mask = resized_masks.sigmoid() < 0.5
        attention_mask = attention_mask.flatten(2)

        # 如果某个 query 把整个 memory 都屏蔽，softmax 会没有有效位置并产生 NaN。
        # 这种少见情况下降级为允许它看完整 feature map，保证训练数值稳定。
        fully_masked = attention_mask.all(dim=-1, keepdim=True)
        attention_mask = attention_mask.masked_fill(fully_masked, False)

        # MultiheadAttention 的 attn_mask 需要把 batch 和 head 合并到第 0 维。
        batch_size, num_queries, num_tokens = attention_mask.shape
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = attention_mask.flatten(0, 1)
        return attention_mask.detach().reshape(batch_size * self.num_heads, num_queries, num_tokens)

    def forward(
        self,
        mask_features: torch.Tensor,
        multi_scale_features: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if len(multi_scale_features) != 3:
            raise ValueError("Mask2FormerQueryDecoder expects exactly 3 multi-scale features.")

        # 预处理三个尺度的 memory：通道投影、加入尺度编码、加入二维位置编码并展平。
        memories: list[torch.Tensor] = []
        memory_positions: list[torch.Tensor] = []
        spatial_sizes: list[tuple[int, int]] = []
        for level_index, feature in enumerate(multi_scale_features):
            projected = self.input_projections[level_index](feature)
            projected = projected + self.level_embed.weight[level_index].view(1, -1, 1, 1)
            position = self.position_embedding(projected)
            memory, memory_pos = self._flatten_memory(projected, position)
            memories.append(memory)
            memory_positions.append(memory_pos)
            spatial_sizes.append(projected.shape[-2:])

        # 将可学习 query 扩展到 batch 维度，得到 [B, Q, C]。
        batch_size = mask_features.shape[0]
        queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = self.query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        predictions: list[dict[str, torch.Tensor]] = []
        # 初始预测只依赖 learnable query，用于构造第 1 层 masked attention 的掩码。
        current_prediction = self._predict_heads(queries, mask_features, output_size)
        for layer_index, layer in enumerate(self.layers):
            # Mask2Former 会在 coarse-to-fine 多尺度 memory 间循环：0,1,2,0,1,2...
            level_index = layer_index % len(memories)
            attention_mask = self._build_attention_mask(
                current_prediction["pred_masks"],
                spatial_sizes[level_index],
            )
            queries = layer(
                queries=queries,
                memory=memories[level_index],
                query_pos=query_pos,
                memory_pos=memory_positions[level_index],
                attention_mask=attention_mask,
            )
            # 每层更新 query 后立即预测一次，既服务下一层 mask attention，也可用于辅助损失。
            current_prediction = self._predict_heads(queries, mask_features, output_size)
            predictions.append(current_prediction)

        # 最后一层作为主输出，前面各层作为 aux_outputs。
        final_prediction = predictions[-1]
        return {
            "pred_logits": final_prediction["pred_logits"],
            "pred_masks": final_prediction["pred_masks"],
            "pred_boxes": final_prediction["pred_boxes"],
            "aux_outputs": predictions[:-1],
        }


class InstanceModel(nn.Module):
    """基于 DINOv3 的田块实例分割网络。

    这个类只负责神经网络前向传播，不在内部实现 Hungarian matching、损失函数、
    指标统计或矢量化后处理。这样同一个模型可以灵活接入不同训练损失和不同推理流程。

    网络主体分为四部分：
    1. ``backbone``：DINOv3/ViT，抽取 4 个中间层特征；
    2. ``pixel_decoder``：DPT/FPN 风格像素解码器，生成高分辨率 mask feature；
    3. ``query_decoder``：Mask2Former 风格实例 decoder，输出 query 级别预测；
    4. 辅助 dense heads：语义、边界、距离图，用于增强像素级监督。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 300,
        decoder_layers: int = 9,
        decoder_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        drop_rate: float = 0.0,
        pretrained_path: str | None = None,
        dino_model_name: str = "vit_large_patch16_dinov3.sat493m",
        dino_pretrained: bool = True,
        dino_out_indices: Sequence[int] | None = None,
        freeze_backbone: bool = False,
        trainable_backbone_blocks: int | None = None,
        return_aux_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.return_aux_outputs = return_aux_outputs

        # 选择 DINO 中需要导出的 4 个 block 特征，并以 features_only 模式创建 backbone。
        out_indices = _as_4tuple(dino_out_indices, dino_model_name)
        self.backbone = timm.create_model(
            dino_model_name,
            pretrained=dino_pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
        )
        # 根据配置冻结整个 backbone，或只放开最后若干个 ViT block 微调。
        self._configure_backbone_finetuning(freeze_backbone, trainable_backbone_blocks)

        # timm feature_info 记录每个输出 level 的通道数，供 pixel decoder 做通道投影。
        feature_channels = self.backbone.feature_info.channels()
        self.pixel_decoder = DptPyramidPixelDecoder(feature_channels, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout2d(drop_rate)
        self.query_decoder = Mask2FormerQueryDecoder(
            num_queries=num_queries,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_ffn_dim,
            dropout=drop_rate,
        )

        # 三个 dense auxiliary head 都接在高分辨率 mask_features 上：
        # semantic_head: 前景/田块区域 logits；
        # boundary_head: 边界二分类 logits；
        # distance_head: 到边界或中心的距离回归图，具体监督由 loss 定义。
        self.semantic_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        self.distance_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.semantic_head.apply(_init_decoder_weights)
        self.boundary_head.apply(_init_decoder_weights)
        self.distance_head.apply(_init_decoder_weights)

        if pretrained_path:
            self._load_weights(pretrained_path)

    def _load_weights(self, pretrained_path: str) -> None:
        """加载 checkpoint 中与当前模型形状兼容的权重。

        训练框架保存权重时可能带有 ``model.`` 或 ``module.`` 前缀；这里会自动清理。
        只加载名称存在且 shape 完全一致的参数，避免因为类别数、query 数或 head 结构变化
        导致加载失败。
        """

        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        current = self.state_dict()
        matched = {}
        for key, value in checkpoint.items():
            # 兼容 LightningModule 的 "model." 前缀和 DataParallel 的 "module." 前缀。
            normalized_key = key.removeprefix("model.").removeprefix("module.")
            if normalized_key in current and current[normalized_key].shape == value.shape:
                matched[normalized_key] = value
        current.update(matched)
        self.load_state_dict(current)

    def _configure_backbone_finetuning(
        self,
        freeze_backbone: bool,
        trainable_backbone_blocks: int | None,
    ) -> None:
        """配置 DINO backbone 的冻结/微调策略。

        参数含义：
        - ``freeze_backbone=True`` 且 ``trainable_backbone_blocks=None``：冻结全部 backbone；
        - ``trainable_backbone_blocks=N``：先冻结全部 backbone，再只解冻最后 N 个 ViT block；
        - ``trainable_backbone_blocks=0``：冻结全部 backbone；
        - 两者都不启用：backbone 全量参与训练。
        """

        if trainable_backbone_blocks is None:
            if freeze_backbone:
                # 完全冻结 DINO，只训练 pixel decoder、query decoder 和各预测头。
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            return

        if trainable_backbone_blocks < 0:
            raise ValueError("trainable_backbone_blocks must be >= 0 or None.")

        # block-wise 微调时先冻结全部参数，再按需解冻尾部 block。
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # timm 的 features_only 包装器通常把原始 ViT 放在 .model 中；若没有则直接用 backbone。
        vit_model = getattr(self.backbone, "model", self.backbone)
        blocks = getattr(vit_model, "blocks", None)
        if blocks is None:
            raise ValueError("The selected DINO backbone does not expose model.blocks for block-wise fine-tuning.")

        num_blocks = len(blocks)
        if trainable_backbone_blocks > num_blocks:
            raise ValueError(
                f"trainable_backbone_blocks={trainable_backbone_blocks} exceeds "
                f"the backbone depth {num_blocks}."
            )

        if trainable_backbone_blocks == 0:
            return

        # 只微调最后 N 个 block：这些层语义更强、任务适配收益更大，参数风险相对可控。
        for block in blocks[-trainable_backbone_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        # 最终 LayerNorm 直接影响输出特征分布，通常和尾部 block 一起放开。
        final_norm = getattr(vit_model, "norm", None)
        if final_norm is not None:
            for parameter in final_norm.parameters():
                parameter.requires_grad = True

    def forward_features(self, images: torch.Tensor | Sequence[torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """只运行 DINO backbone 和 DPT/FPN pixel decoder。

        该方法适合调试特征、可视化 feature pyramid，或在其它模型/损失中复用像素特征。
        返回的 ``mask_features`` 可直接供 query decoder 和 dense auxiliary heads 使用。
        """

        if isinstance(images, (list, tuple)):
            images = _pad_tensor_list(images)

        # backbone 输出 4 个 DINO 中间层特征；pixel_decoder 将其融合成 mask feature 和多尺度 memory。
        features = self.backbone(images)
        pixel_outputs = self.pixel_decoder(features)
        mask_features = self.dropout(pixel_outputs["mask_features"])  # type: ignore[index]
        multi_scale_features = [
            self.dropout(feature)
            for feature in pixel_outputs["multi_scale_features"]  # type: ignore[index]
        ]
        return {
            "mask_features": mask_features,
            "multi_scale_features": multi_scale_features,
            "pyramid": pixel_outputs["pyramid"],
        }

    def forward(self, images: torch.Tensor | Sequence[torch.Tensor]) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """预测田块实例以及像素级辅助结果。

        输入可以是标准 ``Tensor[B, C, H, W]``，也可以是由 Dataset collate 出来的
        ``list[Tensor[C, H_i, W_i]]``。输出字典中的主任务结果是 query 级实例预测，
        辅助结果用于语义区域、边界和距离图监督。
        """

        if isinstance(images, (list, tuple)):
            images = _pad_tensor_list(images)

        # output_size 记录原 batch 的输入空间尺寸，所有输出最终都会插值回这个 H/W。
        output_size = images.shape[-2:]

        # 1. DINO backbone 抽取多层 ViT 特征。
        features = self.backbone(images)
        # 2. DPT/FPN pixel decoder 生成高分辨率 mask_features 和 query decoder 所需多尺度 memory。
        pixel_outputs = self.pixel_decoder(features)
        mask_features = self.dropout(pixel_outputs["mask_features"])  # type: ignore[index]
        multi_scale_features = [
            self.dropout(feature)
            for feature in pixel_outputs["multi_scale_features"]  # type: ignore[index]
        ]

        # 3. Mask2Former query decoder 输出实例类别、实例 mask 和归一化框。
        instance_outputs = self.query_decoder(
            mask_features=mask_features,
            multi_scale_features=multi_scale_features,
            output_size=output_size,
        )

        # 4. 三个 dense auxiliary head 都从 mask_features 预测，并插值回输入图大小。
        semantic_logits = F.interpolate(
            self.semantic_head(mask_features),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        boundary_logits = F.interpolate(
            self.boundary_head(mask_features),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        distance_map = F.interpolate(
            self.distance_head(mask_features),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        # 组装统一输出。损失函数可以按需读取实例预测和辅助 dense 预测。
        outputs: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "pred_logits": instance_outputs["pred_logits"],  # type: ignore[dict-item]
            "pred_masks": instance_outputs["pred_masks"],  # type: ignore[dict-item]
            "pred_boxes": instance_outputs["pred_boxes"],  # type: ignore[dict-item]
            "semantic_logits": semantic_logits,
            "boundary_logits": boundary_logits,
            "distance_map": distance_map,
        }
        if self.return_aux_outputs:
            # aux_outputs 包含除最后一层外的 decoder 中间预测，训练时可加辅助损失。
            outputs["aux_outputs"] = instance_outputs["aux_outputs"]  # type: ignore[dict-item]
        return outputs


if __name__ == "__main__":
    model = InstanceModel(
        dino_model_name="vit_small_patch16_dinov3",
        dino_pretrained=False,
        hidden_dim=64,
        num_queries=20,
        decoder_layers=2,
        decoder_heads=4,
        decoder_ffn_dim=256,
    )
    dummy = torch.randn(2, 3, 256, 256)
    output = model(dummy)
    print("pred_logits:", output["pred_logits"].shape)
    print("pred_masks:", output["pred_masks"].shape)
    print("pred_boxes:", output["pred_boxes"].shape)
    print("semantic_logits:", output["semantic_logits"].shape)
    print("boundary_logits:", output["boundary_logits"].shape)
    print("distance_map:", output["distance_map"].shape)
