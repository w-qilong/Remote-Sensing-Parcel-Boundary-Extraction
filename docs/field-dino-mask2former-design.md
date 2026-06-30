# Field-DINO-Mask2Former 实例分割网络设计

本文档设计一个面向农田边界识别的实例分割网络。目标是将农田边界识别从传统二值语义分割任务，升级为田块级实例分割任务：模型不只判断像素是否属于农田，还要输出每一块独立农田的实例掩膜和边界。

## 1. 任务定义

给定一幅遥感影像 `I`，模型需要预测一组农田实例：

```text
{(mask_i, score_i, optional_box_i)} for i = 1..N
```

其中：

- `mask_i` 是第 `i` 块农田的二值实例掩膜。
- `score_i` 是该实例为有效田块的置信度。
- `optional_box_i` 是该实例的外接框，可用于 Hungarian matching、后处理和大图拼接。

与语义分割相比，实例分割更适合农田边界识别，因为相邻田块通常属于同一语义类别。如果只输出一张“农田/背景”二值图，相邻田块容易粘连；实例分割则显式要求模型把每块田作为独立对象预测。

## 2. 数据基础：FBIS-22M

训练数据使用 FBIS-22M。该数据集以农业田块实例为核心标注目标，包含大规模多区域、多分辨率遥感影像和田块多边形标签，适合训练面向田块实例分割的模型。

项目中已有 `others/visualize_FBIS22M.py`，可以看到当前 FBIS-22M 标签采用 YOLO polygon segmentation 格式：

```text
class_id x1 y1 x2 y2 ... xn yn
```

其中坐标为归一化坐标。训练时建议在 Dataset 中动态完成如下转换：

```text
YOLO polygons
    -> instance masks
    -> instance boxes
    -> union field mask
    -> boundary map
    -> distance map
```

这样可以避免预先生成大量中间标签文件，同时方便进行随机缩放、裁剪、翻转和旋转等几何增强。

## 3. 总体结构

推荐模型命名为 `FieldDinoMask2Former`。

```text
Input image
    |
    v
DINOv3 ViT backbone
    |
    v
Multi-level DINO features
    |
    v
DPT-style reassemble / pixel decoder
    |
    +--------------------------+
    |                          |
    v                          v
Mask2Former query decoder      Dense auxiliary heads
    |                          |
    v                          v
Instance masks/classes/boxes   union mask / boundary / distance
```

核心思想是：

1. 使用 DINOv3 作为强泛化遥感特征提取器。
2. 使用 DPT 风格的重组模块把 ViT token 特征恢复成多尺度图像特征。
3. 使用 Mask2Former 风格的 query decoder 预测田块实例。
4. 保留边界和距离辅助监督，强化田块之间的分割线。

## 4. Backbone：DINOv3 特征提取网络

建议沿用当前项目 `model/dino_dpt.py` 中的 DINOv3/timm 方案。

推荐默认主干：

```text
vit_large_patch16_dinov3.sat493m
```

推荐输出层：

```text
ViT-L/16: block 5, 11, 17, 23
```

设计理由：

- 农田边界依赖纹理、形状、光谱变化、道路、沟渠和作物结构等多种弱语义线索，自监督 ViT 特征通常比纯分类监督特征更适合此类 dense prediction。
- DINO 系列特征具有较好的跨域泛化能力，有利于 FBIS-22M 中不同国家、传感器和分辨率的联合训练。
- 多层 Transformer 特征可以同时提供浅层边界细节、中层区域结构和深层上下文关系。

训练策略建议：

```text
Stage 1: 冻结 DINOv3 backbone，只训练 pixel decoder 和 query decoder
Stage 2: 解冻最后 2-4 个 Transformer block，使用较小学习率微调
Stage 3: 可选引入 LoRA / adapter，以更低代价适配遥感多分辨率域
```

## 5. Pixel Decoder：DPT 重组为多尺度特征

ViT backbone 的不同 block 输出通常具有相同 patch stride，例如 patch size 为 16 时，多个中间特征都是 `H/16 x W/16`。但是实例分割解码器需要多尺度特征，尤其是高分辨率边界细节。

因此使用 DPT-style reassemble，将同 stride token feature map 重组成伪金字塔：

```text
level 0: stride 16 -> stride 4
level 1: stride 16 -> stride 8
level 2: stride 16 -> stride 16
level 3: stride 16 -> stride 32
```

输出建议分为两类：

```text
mask_features:        [B, C, H/4,  W/4]
multi_scale_features: [P3, P4, P5]
```

其中：

- `mask_features` 用于和 query embedding 点积生成最终实例 mask。
- `multi_scale_features` 用于 Mask2Former decoder 的多尺度 cross-attention。

如果后续显存允许，可以把当前 DPT decoder 进一步升级为 deformable attention pixel decoder，以增强跨尺度对齐能力。但第一版实现建议优先复用项目已有 DPT 结构，工程风险更低。

## 6. Query Decoder：Mask2Former 风格实例解码器

实例分割分支采用 query-based mask classification。

推荐配置：

```text
num_queries:      300
hidden_dim:       256
decoder_layers:   9
attention_heads:  8
ffn_dim:          2048
dropout:          0.0
```

每个 learnable query 表示一个候选田块实例。decoder 每层包含：

```text
masked cross-attention
self-attention between queries
feed-forward network
mask prediction
```

每层输出：

```text
class logits: [B, Q, 2]
mask logits:  [B, Q, H/4, W/4]
box pred:     [B, Q, 4] optional
```

最终上采样得到输入分辨率的实例 mask。

设计理由：

- Mask2Former 的 masked attention 让 query 只关注自身预测区域附近的 pixel feature，适合分离相邻田块。
- Query decoder 可以直接预测可变数量实例，不需要 anchor 或传统 proposal。
- 多层辅助输出可以提升训练稳定性，尤其适合 FBIS-22M 中大量小田块、细长田块和密集田块场景。
- 加入 box 分支可以借鉴 Mask DINO 的思想，让 query 同时具备区域定位能力和 mask 生成能力，从而提升 Hungarian matching 稳定性。

## 7. 辅助分支：union mask、boundary、distance

不建议完全丢弃当前 `DinoDpt` 中已有的 dense auxiliary heads。田块实例分割的关键难点是相邻实例粘连，边界和距离监督可以补充 query decoder 的不足。

建议保留三个辅助输出：

```text
semantic_field_mask: 所有田块实例的并集
boundary_map:        田块边界二分类图
distance_map:        距离边界或实例中心的距离图
```

设计理由：

- `semantic_field_mask` 强化整体农田区域识别，避免 query decoder 漏检大块区域。
- `boundary_map` 显式监督田块之间的共享边界，减少实例粘连。
- `distance_map` 提供几何结构先验，帮助模型区分田块内部、边界附近和背景。

因此最终网络不是纯 Mask2Former，而是：

```text
DINOv3 + DPT pixel decoder + Mask2Former instance head + boundary-aware auxiliary heads
```

## 8. 损失函数

主实例分割损失采用 DETR/Mask2Former 风格的 Hungarian matching。

### 8.1 匹配代价

对每张图，将 `Q` 个预测和 `M` 个真实田块实例做二分图匹配：

```text
cost = λ_cls  * CE(class)
     + λ_mask * BCE/Focal(mask)
     + λ_dice * Dice(mask)
     + λ_box  * L1(box)
     + λ_giou * GIoU(box)
```

推荐初始权重：

```text
λ_cls  = 2.0
λ_mask = 5.0
λ_dice = 5.0
λ_box  = 2.0
λ_giou = 2.0
```

如果第一版暂不实现 box 分支，可以移除 `λ_box` 和 `λ_giou`。

### 8.2 总损失

```text
L = L_instance
  + 0.5 * L_union_mask
  + 1.0 * L_boundary
  + 0.2 * L_distance
  + L_aux_decoder_layers
```

其中：

- `L_instance`: matched query 的 class CE、mask BCE/Focal、mask Dice、可选 box/GIoU。
- `L_union_mask`: 所有 GT 实例并集的 BCE + Dice。
- `L_boundary`: 边界 Focal 或 Dice loss，用于处理边界类别极度不平衡。
- `L_distance`: SmoothL1 或 MSE。
- `L_aux_decoder_layers`: decoder 每层辅助输出的实例损失。

### 8.3 Point-based mask loss

对于高分辨率训练，建议采用 point sampling 计算 mask loss：

```text
从不确定区域和随机区域采样若干点
只在采样点上计算 BCE/Focal + Dice
```

这样可以显著降低显存占用，同时让模型重点学习边界附近的不确定像素。

## 9. FBIS-22M Dataset 设计建议

新增数据集类建议命名为：

```text
Fbis22mInstanceDataset
```

单个样本返回：

```python
{
    "file_name": str,
    "image": Tensor[C, H, W],
    "instances": {
        "masks": Tensor[N, H, W],
        "boxes": Tensor[N, 4],
        "labels": Tensor[N],
    },
    "semantic_mask": Tensor[1, H, W],
    "boundary": Tensor[1, H, W],
    "distance": Tensor[1, H, W],
    "resolution": Optional[float],
}
```

推荐数据增强：

```text
RandomResize: 0.5 - 2.0
RandomCrop: 512 or 768
HorizontalFlip
VerticalFlip
RandomRotate90
ColorJitter
GaussianBlur / Sharpen
Resolution-aware augmentation
```

由于 FBIS-22M 包含多传感器、多分辨率影像，建议解析文件名或 metadata 得到空间分辨率，并加入 resolution embedding：

```text
resolution value -> MLP -> resolution embedding
```

该 embedding 可以加到 query embedding 或 pixel feature 中，帮助模型区分不同地面采样距离下的纹理尺度。

## 10. 推理与后处理

模型推理输出 `Q` 个实例候选：

```text
pred_logits: [B, Q, 2]
pred_masks:  [B, Q, H, W]
pred_boxes:  [B, Q, 4] optional
```

后处理流程：

```text
1. 过滤 no-object query 和低置信度 query
2. 对 mask 做 sigmoid 和阈值化
3. 处理 query 之间的重叠像素，保留 score 更高的实例
4. 删除面积过小、形状异常或过碎的实例
5. 可选使用 boundary_map 修正相邻实例粘连
6. binary mask -> polygon
7. polygon simplify / topology clean
```

大图推理建议采用滑窗：

```text
tile_size: 512 or 768
overlap:   64 or 128
```

跨 tile 合并可结合：

```text
mask IoU
box IoU
polygon overlap
boundary distance
```

## 11. 推荐默认实验配置

```text
model_name: field_dino_mask2former
backbone: vit_large_patch16_dinov3.sat493m
dino_out_indices: 5 11 17 23
hidden_dim: 256
num_queries: 300
decoder_layers: 9
crop_size: 512
batch_size: 根据显存调整，建议从 2-8 开始
optimizer: AdamW
lr_decoder: 1e-4
lr_backbone: 1e-5
weight_decay: 1e-4
precision: bf16-mixed or 16-mixed
epochs: 50-100
```

如果显存有限，可先使用：

```text
backbone: vit_small_patch16_dinov3
hidden_dim: 128
num_queries: 100 or 200
crop_size: 512
```

## 12. 与当前项目的衔接

当前 `model/dino_dpt.py` 已经实现：

- DINOv3 backbone 加载。
- 多层 ViT 特征读取。
- DPT-style feature reassemble。
- dense mask、edge、distance 三头输出。

因此后续实现可以分三步推进：

```text
Step 1: 新增 FBIS-22M instance dataset，支持 polygon -> masks/boxes/boundary/distance
Step 2: 新增 FieldDinoMask2Former 模型，复用 DinoDpt 的 backbone 和 DPT decoder
Step 3: 新增 Hungarian matcher 和实例分割 loss，接入 Lightning 训练流程
```

工程上建议保留 `DinoDpt` 作为语义分割基线，新建文件：

```text
model/field_dino_mask2former.py
data/fbis22m_instance_dataset.py
losses/instance_loss.py
```

这样可以避免破坏当前 FTW/HBGNet 风格训练流程。

## 13. 参考方向

- Mask2Former: masked-attention mask classification for universal image segmentation.
- Mask DINO: unified query-based detection and segmentation framework.
- DINOv3: self-supervised vision foundation model with strong dense feature transfer.
- Delineate Anything / FBIS-22M: resolution-agnostic agricultural field boundary delineation using large-scale field instance supervision.

## 14. 总结

推荐的最终模型是：

```text
Field-DINO-Mask2Former
= DINOv3 backbone
+ DPT-style pixel decoder
+ Mask2Former query instance decoder
+ boundary/distance auxiliary dense heads
```

该设计将 DINOv3 的强泛化遥感表征、DPT 的高分辨率边界恢复能力、Mask2Former 的 query-based 实例分割能力结合起来。对于农田边界识别，它比单纯二值语义分割更能处理相邻田块粘连、小田块密集、跨分辨率泛化和最终矢量化制图等关键问题。
