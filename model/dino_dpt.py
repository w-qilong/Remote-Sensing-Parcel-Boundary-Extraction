"""DINOv3 + DPT decoder for dense semantic segmentation.

本模块遵循项目的动态加载约定：文件名 ``dino_dpt.py`` 对应类名
``DinoDpt``，因此可以通过 ``--model_name dino_dpt`` 直接实例化。

DINOv3/ViT 的中间层特征默认都是 stride=16 的 token feature map。这里借鉴
DPT 的思路：从多个 Transformer block 取特征，先 reassemble 成不同空间尺度的
伪金字塔，再用卷积式 fusion decoder 逐级融合，最后上采样到输入图像同尺寸。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover - import-time guard for clearer errors.
    raise ImportError("DinoDpt requires timm. Install project dependencies first.") from exc


def _init_decoder_weights(module: nn.Module) -> None:
    """初始化新增的 DPT 解码器层。

    注意：这个函数只应该作用在我们自己新建的 decoder/head 上，不要对
    ``self.backbone`` 调用，否则会覆盖 DINOv3 已加载的预训练权重。
    """

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        # 卷积层采用 Kaiming 初始化，适合后面接 ReLU 的卷积解码器。
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        # 归一化层初始化为恒等变换：不缩放、不平移。
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _default_out_indices(model_name: str) -> tuple[int, int, int, int]:
    """为常见 DINOv3/ViT 规格选择 4 个中间层输出。

    DPT 的核心做法之一是融合 Transformer 不同深度的表示：
    - 浅层：更偏局部纹理、边界；
    - 中层：逐渐形成物体部件和区域；
    - 深层：语义更强、全局关系更充分。

    timm 的 ViT feature extractor 使用 block 下标作为 ``out_indices``。
    不同规模模型 block 数不同，所以这里按模型深度取大致均匀分布的 4 层。
    """

    name = model_name.lower()
    if "7b" in name:
        return (9, 19, 29, 39)
    if "huge" in name:
        return (7, 15, 23, 31)
    if "large" in name:
        return (5, 11, 17, 23)
    return (2, 5, 8, 11)


def _make_reassemble_layer(index: int, channels: int) -> nn.Module:
    """把 ViT 同尺度特征重组为 DPT 风格的伪多尺度特征。

    DINOv3 ViT 的中间特征通常都是 patch 网格，空间 stride 相同，例如输入
    256x256、patch=16 时，每层都是 16x16。CNN 分割解码器则更习惯接收
    1/4、1/8、1/16、1/32 这类金字塔特征。

    因此这里不改变语义来源，只改变空间尺度：
    - index 0: stride 16 -> stride 4，用反卷积放大 4 倍；
    - index 1: stride 16 -> stride 8，用反卷积放大 2 倍；
    - index 2: 保持 stride 16；
    - index 3: stride 16 -> stride 32，用卷积下采样 2 倍。

    这样后续 decoder 可以像处理 CNN feature pyramid 一样逐级融合。
    """

    if index == 0:
        # 最浅层特征需要保留更多边界细节，因此提升到最高的 decoder 分辨率。
        return nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    if index == 1:
        # 第二层作为中高分辨率 skip，放大 2 倍即可。
        return nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    if index == 2:
        # 第三层作为原始 patch 网格尺度，直接传递。
        return nn.Identity()
    # 最深层语义最强，用 stride=2 卷积形成最低分辨率起点。
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


class ConvNormAct(nn.Module):
    """解码器中的基础卷积块：Conv2d -> BatchNorm -> ReLU。

    ``kernel_size=1`` 时主要用于通道投影；``kernel_size=3`` 时用于融合局部邻域。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvUnit(nn.Module):
    """DPT 风格的残差卷积单元。

    Transformer 提供了全局上下文，但 token 特征 reshape 回图像后，仍需要卷积
    补充局部平滑和边界细化。残差结构让该模块可以在必要时近似恒等映射，训练更稳。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入/输出形状均为 [B, C, H, W]，只在局部空间邻域内细化特征。
        residual = self.conv2(self.conv1(x))
        residual = self.bn(residual)
        return self.act(x + residual)


class FeatureFusionBlock(nn.Module):
    """DPT 解码器的一级融合模块。

    输入 ``x`` 是当前较低分辨率、语义更强的特征；``skip`` 是来自 reassemble
    后的较高分辨率特征。融合逻辑是：
    1. 如有 skip，先把 x 对齐到 skip 的空间尺寸；
    2. 对 skip 做残差卷积细化后与 x 相加；
    3. 再做一次残差卷积；
    4. 上采样 2 倍，把结果交给下一层更高分辨率的融合块。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.skip_unit = ResidualConvUnit(channels)
        self.out_unit = ResidualConvUnit(channels)
        self.out_conv = ConvNormAct(channels, channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                # 对齐空间尺寸后才能逐像素相加；这里不改变通道数。
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            # 相加融合相当于 FPN/DPT 中的 skip connection，保留细节同时注入深层语义。
            x = x + self.skip_unit(skip)
        x = self.out_unit(x)
        # 每经过一个 fusion block，空间分辨率提升一倍。
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class DptFeatureDecoder(nn.Module):
    """把 DINOv3 多层特征解码为原图尺寸的 dense feature map。

    输入：
        ``features``: timm DINOv3 backbone 输出的多层特征列表，每个元素形如
        ``[B, C_i, H_patch, W_patch]``。

    输出：
        ``[B, hidden_dim, H_img, W_img]``，即每个原图像素都有一个 hidden_dim 维特征。

    这里分三步：
    1. ``projections``：把不同 backbone 通道统一到 ``hidden_dim``；
    2. ``reassemble``：把同 stride 的 ViT 特征变成伪多尺度金字塔；
    3. ``fusion``：从最低分辨率向最高分辨率逐级融合，最后插值回原图大小。
    """

    def __init__(self, in_channels: Sequence[int], hidden_dim: int) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("DptFeatureDecoder needs at least two DINO feature levels.")

        # ViT 不同模型的通道数可能是 384/768/1024 等；先统一为 decoder 宽度。
        self.projections = nn.ModuleList(
            [ConvNormAct(channels, hidden_dim, kernel_size=1) for channels in in_channels]
        )
        # 将 4 个同分辨率 token feature map 变成类似 CNN 的多尺度 pyramid。
        self.reassemble = nn.ModuleList(
            [_make_reassemble_layer(idx, hidden_dim) for idx in range(len(in_channels))]
        )
        # 每个 pyramid level 对应一个 fusion block，从深到浅反向使用。
        self.fusion = nn.ModuleList([FeatureFusionBlock(hidden_dim) for _ in in_channels])
        self.final = nn.Sequential(
            ResidualConvUnit(hidden_dim),
            ConvNormAct(hidden_dim, hidden_dim),
        )
        self.apply(_init_decoder_weights)

    def forward(self, features: Sequence[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")

        # 对每层 DINO 特征执行：通道投影 -> 空间尺度重组。
        # 若输入 256x256 且 patch=16，典型 pyramid 尺寸约为：
        # [64x64, 32x32, 16x16, 8x8]。
        pyramid = [
            reassemble(project(feature))
            for feature, project, reassemble in zip(features, self.projections, self.reassemble)
        ]

        # 从最低分辨率、语义最强的特征开始，逐级融合更高分辨率 skip。
        x = pyramid[-1]
        for idx, fusion_block in enumerate(reversed(self.fusion)):
            skip_index = len(pyramid) - 1 - idx
            skip = pyramid[skip_index - 1] if skip_index > 0 else None
            x = fusion_block(x, skip)

        # fusion 后的尺寸通常已经高于/接近输入尺寸，最终显式插值到原图 H,W，
        # 保证输出可以直接用于逐像素语义分割监督。
        x = self.final(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class DinoDpt(nn.Module):
    """DINOv3 backbone with a DPT decoder for FTW-style dense prediction.

    ``forward_features`` returns the same-size dense feature map ``[B, hidden_dim, H, W]``.
    ``forward`` adds segmentation heads. With ``return_aux_outputs=True`` it matches the
    existing ``LossF`` contract: ``[mask_logits, edge_log_probs, distance_map]``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        hidden_dim: int = 256,
        drop_rate: float = 0.2,
        pretrained_path: str | None = None,
        return_aux_outputs: bool = True,
        dino_model_name: str = "vit_large_patch16_dinov3.sat493m",
        dino_pretrained: bool = True,
        dino_out_indices: Sequence[int] | None = None,
        freeze_backbone: bool = False,
        trainable_backbone_blocks: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.return_aux_outputs = return_aux_outputs

        # 如果用户没有手动指定 dino_out_indices，就根据模型规模选 4 个中间 block。
        out_indices = tuple(dino_out_indices) if dino_out_indices is not None else _default_out_indices(dino_model_name)

        # timm 会把 ViT token 自动 reshape 成 [B, C, H_patch, W_patch] 的特征图。
        # pretrained=True 时会自动加载对应 DINOv3 权重；首次运行可能需要联网下载。
        self.backbone = timm.create_model(
            dino_model_name,
            pretrained=dino_pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
        )
        # 控制 DINOv3 主干是否参与微调：
        # - trainable_backbone_blocks=None 时，沿用 freeze_backbone 的旧逻辑；
        # - trainable_backbone_blocks=N 时，冻结整个 backbone，仅解冻最后 N 个 ViT block；
        # - trainable_backbone_blocks=0 时，相当于冻结整个 backbone。
        self._configure_backbone_finetuning(freeze_backbone, trainable_backbone_blocks)

        # feature_info.channels() 返回每个 out_indices 对应的通道数，例如 large 为 1024。
        feature_channels = self.backbone.feature_info.channels()
        self.decoder = DptFeatureDecoder(feature_channels, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout2d(drop_rate)

        # 三个输出头兼容项目现有 LossF：
        # mask_head: 二值田块/背景 mask logits，给 BCEWithLogits + Dice 使用；
        # edge_head: 边界二分类 log-prob，给 NLLLoss 使用；
        # distance_head: 距离图回归，给 MSE 使用。
        self.mask_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.distance_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        # 只初始化新增 head；decoder 已在 DptFeatureDecoder 内初始化。
        self.mask_head.apply(_init_decoder_weights)
        self.edge_head.apply(_init_decoder_weights)
        self.distance_head.apply(_init_decoder_weights)

        if pretrained_path:
            self._load_weights(pretrained_path)

    def _load_weights(self, pretrained_path: str) -> None:
        """从本地 checkpoint 加载名称和形状都匹配的权重。

        这里使用宽松加载，原因是训练保存的 checkpoint 可能来自 Lightning，
        key 前面会带 ``model.`` 或 ``module.``；也可能只想加载 decoder/head 的一部分。
        形状不匹配的权重会被跳过，避免因为类别数或 hidden_dim 改变而报错。
        """

        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        current = self.state_dict()
        matched = {}
        for key, value in checkpoint.items():
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
        """配置 DINOv3 backbone 的冻结/微调范围。

        ``timm.create_model(..., features_only=True)`` 返回的是 ``FeatureGetterNet``，
        真正的 ViT 通常包在 ``self.backbone.model`` 中。DINOv3 的 Transformer
        层位于 ``model.blocks``，所以这里可以按 block 数量精确控制最后几层可训练。

        Args:
            freeze_backbone: 为 True 且未指定 ``trainable_backbone_blocks`` 时，
                冻结整个 backbone。
            trainable_backbone_blocks: 指定可微调的最后 N 个 Transformer block。
                例如 ``4`` 表示只训练最后 4 个 block，前面的 patch embedding、
                position/rope 以及更早 block 都保持冻结。
        """

        if trainable_backbone_blocks is None:
            if freeze_backbone:
                # 兼容原有参数：只要 freeze_backbone=True，就冻结整个主干。
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            return

        if trainable_backbone_blocks < 0:
            raise ValueError("trainable_backbone_blocks must be >= 0 or None.")

        # 先冻结整个 DINOv3 backbone，再按需解冻尾部 block。
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

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

        # 只解冻最后 N 个 Transformer block。前面的 block 和 patch embedding 保持冻结。
        for block in blocks[-trainable_backbone_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        # 如果模型带最终归一化层，也一起微调；它通常紧贴最后 block 的输出分布。
        final_norm = getattr(vit_model, "norm", None)
        if final_norm is not None:
            for parameter in final_norm.parameters():
                parameter.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回与输入图像同尺寸的密集特征图。

        Args:
            x: 输入图像，形状为 ``[B, C, H, W]``。

        Returns:
            dense_features: 形状为 ``[B, hidden_dim, H, W]``。这就是你后续做
            dense semantic segmentation 时最核心的逐像素特征。
        """
        input_size = x.shape[-2:]
        # DINOv3 输出多个 block 的 patch feature map，例如 256x256 输入对应 16x16。
        features = self.backbone(x)
        # DPT decoder 将 patch feature map 融合并恢复到输入图像分辨率。
        dense_features = self.decoder(features, output_size=input_size)
        return self.dropout(dense_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """前向预测。

        默认返回三路输出以兼容 ``LossF``。如果只想要主分割结果，可以在初始化时
        设置 ``return_aux_outputs=False``，此时只返回 ``mask_logits``。
        """
        dense_features = self.forward_features(x)

        # 主分割输出不做 sigmoid；LossF 内部使用 BCEWithLogitsLoss，会自己处理 logits。
        mask_logits = self.mask_head(dense_features)
        if not self.return_aux_outputs:
            return mask_logits

        # 边界分支输出 log_softmax，因为 LossMulti 内部使用 NLLLoss。
        edge_log_probs = F.log_softmax(self.edge_head(dense_features), dim=1)
        # 距离图分支是回归任务，直接输出实数图。
        distance_map = self.distance_head(dense_features)
        return [mask_logits, edge_log_probs, distance_map]


if __name__ == "__main__":
    model = DinoDpt(
        dino_model_name="vit_small_patch16_dinov3",
        dino_pretrained=False,
        hidden_dim=64,
        return_aux_outputs=True,
    )
    dummy_input = torch.randn(2, 3, 256, 256)
    dense = model.forward_features(dummy_input)
    outputs = model(dummy_input)
    print("dense feature shape:", dense.shape)
    print("mask shape:", outputs[0].shape)
    print("edge shape:", outputs[1].shape)
    print("distance shape:", outputs[2].shape)
