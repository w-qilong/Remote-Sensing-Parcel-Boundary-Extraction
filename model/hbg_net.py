"""基于 PVT-v2 主干的自包含 HBGNet 模型。

本文件遵循项目里的动态加载约定：文件名是 ``hbg_net.py``，可加载类名是
``HbgNet``，因此 ``MInterface`` 可以通过 ``--model_name hbg_net`` 自动实例化。
模型内部同时包含 PVT-v2 风格编码器、边界引导解码器和三个训练输出头，
不再依赖原项目中拆散的 ``pvtv2.py`` 等外部模型文件。
"""

from __future__ import annotations

from functools import partial
import math

import torch
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
import torch.nn.functional as F


def _init_vit_weights(module: nn.Module) -> None:
    """按 PVT-v2/ViT 常用规则初始化 Linear、LayerNorm 和 Conv2d。"""

    # PVT/ViT 系列常用截断正态初始化 Linear 权重，偏置置零。
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    # LayerNorm 默认保持恒等缩放：weight=1, bias=0。
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    # Conv2d 采用类似 Kaiming 的初始化，fan_out 会考虑分组卷积的 groups。
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


class DepthwiseTokenConv(nn.Module):
    """PVT MLP 内部使用的深度可分离卷积。

    PVT block 中的 token 形状是 ``(batch, tokens, channels)``。该层会临时恢复
    空间特征图，用低成本 depthwise 卷积注入局部图像上下文，再转回 token 序列。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 输入 x 为 token 序列 [B, N, C]，其中 N = H * W。
        batch, _, channels = x.shape
        # depthwise conv 需要 2D 特征图格式，先恢复成 [B, C, H, W]。
        x = x.transpose(1, 2).view(batch, channels, height, width)
        x = self.conv(x)
        # 卷积后再展平回 Transformer 使用的 token 格式 [B, N, C]。
        return x.flatten(2).transpose(1, 2)


class PVTMlp(nn.Module):
    """PVT-v2 编码器 block 中的前馈网络。"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthwiseTokenConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(_init_vit_weights)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 先升维到 hidden_features，再通过 depthwise conv 注入局部空间信息。
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        # 投影回 out_features，保持与残差分支相同的通道维度。
        x = self.fc2(x)
        return self.drop(x)


class SpatialReductionAttention(nn.Module):
    """带可选 K/V 空间降采样的多头自注意力。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(_init_vit_weights)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        # Q 保持原始 token 数量，用于给每个位置生成注意力查询。
        q = self.q(x).reshape(batch, tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # PVT 的空间降采样注意力：只对 K/V 做空间降采样，降低注意力矩阵规模。
            reduced = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).reshape(batch, channels, -1).permute(0, 2, 1)
            reduced = self.norm(reduced)
            kv_source = reduced
        else:
            kv_source = x

        kv = self.kv(kv_source).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        # 标准 scaled dot-product attention，输出形状恢复为 [B, N, C]。
        attn = (q @ key.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ value).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class PVTBlock(nn.Module):
    """单个 PVT-v2 编码器 block，包含注意力、MLP 和残差路径。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        norm_layer: type[nn.Module],
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, qkv_bias, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = PVTMlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.apply(_init_vit_weights)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Pre-Norm Transformer 结构：Norm -> Attention/MLP -> DropPath -> 残差相加。
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    """卷积式重叠 patch embedding，用于保留局部边界上下文。"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_vit_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # 卷积 patch embedding 输出为 [B, C, H', W']。
        x = self.proj(x)
        _, _, height, width = x.shape
        # Transformer block 接收 token 序列，因此展平空间维度为 [B, H'W', C]。
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class PyramidVisionTransformerV2(nn.Module):
    """PVT-v2 主干网络，返回四个尺度的空间特征图。"""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        embed_dims: tuple[int, int, int, int] = (64, 128, 320, 512),
        num_heads: tuple[int, int, int, int] = (1, 2, 5, 8),
        mlp_ratios: tuple[int, int, int, int] = (8, 8, 4, 4),
        depths: tuple[int, int, int, int] = (3, 4, 6, 3),
        sr_ratios: tuple[int, int, int, int] = (8, 4, 2, 1),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 四个阶段逐步降采样，默认输出 stride 约为 4、8、16、32 的金字塔特征。
        self.patch_embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(img_size, 7, 4, in_channels, embed_dims[0]),
                OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1]),
                OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2]),
                OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3]),
            ]
        )

        # DropPath 按 block 深度线性递增，越深的层随机深度越强。
        drop_paths = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        blocks: list[nn.ModuleList] = []
        norms: list[nn.Module] = []
        for stage_idx, depth in enumerate(depths):
            stage_blocks = nn.ModuleList(
                [
                    PVTBlock(
                        dim=embed_dims[stage_idx],
                        num_heads=num_heads[stage_idx],
                        mlp_ratio=mlp_ratios[stage_idx],
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_paths[cursor + block_idx],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[stage_idx],
                    )
                    for block_idx in range(depth)
                ]
            )
            cursor += depth
            blocks.append(stage_blocks)
            norms.append(norm_layer(embed_dims[stage_idx]))
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList(norms)
        self.apply(_init_vit_weights)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for patch_embed, blocks, norm in zip(self.patch_embeds, self.blocks, self.norms):
            batch = x.shape[0]
            # 每个 stage 都先进行重叠 patch embedding，再堆叠若干 PVTBlock。
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            # 解码器使用 CNN 格式特征，因此将 [B, N, C] 还原为 [B, C, H, W]。
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)
        return features


class ConvBNReLU(nn.Module):
    """解码器中复用的小型 Conv-BN-ReLU 模块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """带通道对齐 skip 分支的双卷积残差块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主分支学习局部上下文，skip 分支负责通道对齐并保留输入信息。
        return self.main(x) + self.skip(x)


class SpatialSelfAttention(nn.Module):
    """DANet 风格的长程空间注意力模块。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced_channels = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        # 将每个空间位置视为一个节点，计算任意两位置之间的相似度。
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = torch.bmm(query, key).softmax(dim=-1)
        # 用空间注意力重新聚合 value，gamma 初始为 0，训练初期近似恒等映射。
        value = self.value(x).view(batch, channels, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return x + self.gamma * out


class NearLongFusion(nn.Module):
    """融合局部残差上下文和长程空间注意力。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.long_context = SpatialSelfAttention(in_channels)
        self.local_context = ResidualBlock(in_channels, out_channels)
        self.fuse = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 拼接长程注意力分支和局部残差分支，再用 1x1 卷积压回 out_channels。
        return self.fuse(torch.cat([self.long_context(x), self.local_context(x)], dim=1))


class CBAM(nn.Module):
    """用于多尺度特征细化的通道注意力和空间注意力。"""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力：最大池化和平均池化共享同一个 MLP，融合后生成通道权重。
        channel_weight = self.channel_mlp(self.max_pool(x)) + self.channel_mlp(self.avg_pool(x))
        x = x * self.sigmoid(channel_weight)
        # 空间注意力：沿通道维取 max/mean，学习每个空间位置的重要性。
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.spatial_conv(torch.cat([max_map, avg_map], dim=1)))


class SpatialGroupEnhance(nn.Module):
    """辅助输出头之前使用的空间组增强模块。"""

    def __init__(self, groups: int = 32) -> None:
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels % self.groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={self.groups}")

        # 将通道分成 groups 组，在每组内部估计空间响应。
        grouped = x.view(batch * self.groups, -1, height, width)
        # 每组特征与其全局平均响应相乘，突出与全局语义一致的位置。
        response = (grouped * self.avg_pool(grouped)).sum(dim=1, keepdim=True)
        response = response.view(batch * self.groups, -1)
        # 对每组空间响应做标准化，避免不同样本/组之间尺度差异过大。
        response = response - response.mean(dim=1, keepdim=True)
        response = response / (response.std(dim=1, keepdim=True) + 1e-5)
        response = response.view(batch, self.groups, height, width)
        # 每个 group 拥有可学习的缩放和平移参数，再经过 sigmoid 得到空间权重。
        response = response * self.weight + self.bias
        response = response.view(batch * self.groups, 1, height, width)
        grouped = grouped * self.sigmoid(response)
        return grouped.view(batch, channels, height, width)


class LaplaceConv2d(nn.Module):
    """以 Laplace 核初始化、训练中可学习的边缘提取器。"""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Laplace 核用于强调边缘/轮廓。这里设为可学习参数，训练时可继续更新。
        kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(out_channels, in_channels, 1, 1)
        self.conv.weight = nn.Parameter(kernel)
        nn.init.zeros_(self.conv.bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BoundaryGuidedModule(nn.Module):
    """用边界特征作为语义特征的空间门控。"""

    def __init__(self, edge_channels: int, semantic_channels: int, out_channels: int) -> None:
        super().__init__()
        self.edge_proj = nn.Conv2d(edge_channels, out_channels, kernel_size=1)
        self.semantic_proj = nn.Conv2d(semantic_channels, out_channels, kernel_size=1)

    def forward(self, edge: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        # edge_proj 后沿通道取最大值，得到单通道边界置信图作为空间门控。
        edge_weight, _ = torch.max(self.edge_proj(edge), dim=1, keepdim=True)
        semantic = self.semantic_proj(semantic)
        # 边界门控增强语义特征，同时保留原始 semantic，避免边缘噪声完全抑制特征。
        return semantic * torch.sigmoid(edge_weight) + semantic


class MultiScaleFusion(nn.Module):
    """使用多个感受野分支融合田块语义特征。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0),
                ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1),
                ConvBNReLU(in_channels, out_channels, kernel_size=7, padding=3),
                ConvBNReLU(in_channels, out_channels, kernel_size=11, padding=5),
            ]
        )
        self.squeeze = ConvBNReLU(out_channels * 4, out_channels, kernel_size=1, padding=0)
        self.attention = CBAM(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行卷积分支覆盖 1/3/7/11 不同感受野，拼接后统一压缩通道。
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        # CBAM 在多尺度融合后进一步进行通道和空间重标定。
        return self.attention(self.squeeze(x))


class HbgNet(nn.Module):
    """HBGNet 田块分割模型。

    默认 ``return_aux_outputs=True`` 时返回 ``[mask_logits, edge_log_probs,
    distance_map]``，供 ``LossF`` 同时监督掩膜、边界和距离图。设置为 ``False`` 时，
    仅返回主掩膜 logits，便于复用通用单损失训练路径。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 256,
        drop_rate: float = 0.4,
        pretrained_path: str | None = None,
        return_aux_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.return_aux_outputs = return_aux_outputs
        self.drop = nn.Dropout2d(drop_rate)

        # 主干网络提取四级金字塔语义特征。
        self.backbone = PyramidVisionTransformerV2(img_size=img_size, in_channels=in_channels)
        if pretrained_path:
            self._load_backbone_weights(pretrained_path)

        # 边界分支：先用 Laplace 初始化卷积提取低层边缘，再用残差块升维。
        self.edge_laplace = LaplaceConv2d(in_channels=in_channels, out_channels=1)
        self.edge_stem = ResidualBlock(1, 32)
        self.edge_attention = CBAM(64)
        self.edge_reduce = nn.Conv2d(64, 16, kernel_size=1)

        # 对每一级 PVT 特征分别融合局部上下文和长程空间上下文。
        self.fuse_stage1 = NearLongFusion(64, 32)
        self.fuse_stage2 = NearLongFusion(128, 64)
        self.fuse_stage3 = NearLongFusion(320, 128)
        self.fuse_stage4 = NearLongFusion(512, 256)

        # 高层语义特征用边界特征做引导，统一投影到 16 通道后再拼接。
        self.boundary_stage2 = BoundaryGuidedModule(64, 64, 16)
        self.boundary_stage3 = BoundaryGuidedModule(64, 128, 16)
        self.boundary_stage4 = BoundaryGuidedModule(64, 256, 16)
        self.multi_fusion = MultiScaleFusion(64, 64)

        # 三个输出头：主分割 mask、辅助边缘分类、辅助距离图回归。
        self.mask_head = nn.Conv2d(64, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.distance_head = nn.Conv2d(64, 1, kernel_size=1)
        self.output_enhance = SpatialGroupEnhance(groups=32)

    def _load_backbone_weights(self, pretrained_path: str) -> None:
        """从 checkpoint 中只加载名称和形状都匹配的 PVT-v2 主干权重。"""

        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        current = self.backbone.state_dict()
        # 只加载名称匹配且 shape 一致的权重，避免分类头或结构差异导致加载失败。
        matched = {
            key.replace("backbone.", ""): value
            for key, value in checkpoint.items()
            if key.replace("backbone.", "") in current and current[key.replace("backbone.", "")].shape == value.shape
        }
        current.update(matched)
        self.backbone.load_state_dict(current)

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # 将 x 双线性插值到 reference 的空间分辨率，用于多尺度特征对齐。
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        input_size = x.shape[-2:]

        # 1. 边界分支从原图提取边缘先验，输出 [B, 32, H, W]。
        edge = self.edge_stem(self.edge_laplace(x))
        # 2. PVT 主干输出四级特征，并在训练时通过 Dropout2d 做正则化。
        stage1, stage2, stage3, stage4 = [self.drop(feature) for feature in self.backbone(x)]

        # 3. 每级特征分别经过近邻局部 + 远程注意力融合模块。
        stage1 = self.fuse_stage1(stage1)
        stage2 = self.fuse_stage2(stage2)
        stage3 = self.fuse_stage3(stage3)
        stage4 = self.fuse_stage4(stage4)

        # 4. stage1 分辨率最高，先上采样回输入大小，与边缘分支拼接细化边界。
        stage1 = F.interpolate(stage1, size=input_size, mode="bilinear", align_corners=True)
        edge = self.edge_attention(torch.cat([edge, stage1], dim=1))
        edge_small = self.edge_reduce(edge)

        # 5. 将高层语义特征对齐到边界特征分辨率，并用边界门控增强。
        stage2 = self._resize_like(stage2, edge)
        stage3 = self._resize_like(stage3, edge)
        stage4 = self._resize_like(stage4, edge)
        guided_stage2 = self.boundary_stage2(edge, stage2)
        guided_stage3 = self.boundary_stage3(edge, stage3)
        guided_stage4 = self.boundary_stage4(edge, stage4)

        # 6. 拼接边界低维特征和三层边界引导语义特征，做多尺度融合与组增强。
        fused = self.multi_fusion(torch.cat([edge_small, guided_stage2, guided_stage3, guided_stage4], dim=1))
        fused = self.output_enhance(fused)

        # 7. 输出主分割 logits；辅助输出用于自定义多任务训练。
        mask_logits = self.mask_head(fused)
        edge_log_probs = F.log_softmax(self.edge_head(edge), dim=1)
        distance_map = self.distance_head(fused)
        if self.return_aux_outputs:
            return [mask_logits, edge_log_probs, distance_map]
        return mask_logits

if __name__ == "__main__":
    # 这个模块主要作为模型定义，通常不直接运行。但这里提供一个简单的 smoke test。
    model = HbgNet(in_channels=3, num_classes=2, img_size=256)
    dummy_input = torch.randn(2, 3, 256, 256)  # batch_size=2, RGB image, 256x256
    output = model(dummy_input)
    print("mask shape:", output[0].shape)  # 预期 [2, 1, 256, 256]
    print("edge shape:", output[1].shape)  # 预期 [2, 2, 256, 256]
    print("distance shape:", output[2].shape)  # 预期 [2, 1, 256, 256]
