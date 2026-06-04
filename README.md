# Remote Sensing Parcel Boundary Extraction

本项目用于遥感影像中的田块/地块边界提取。代码基于 PyTorch Lightning 组织训练流程，默认使用 FTW（Fields of The World）数据集，并实现了面向田块分割的 HBGNet 风格多任务模型。

仓库只保存代码、测试和文档，不保存真实数据、日志和 checkpoint。`ftw_data/` 是本地数据目录，已经加入 `.gitignore`，不会被推送到 GitHub。

## 项目功能

- 使用 FTW 数据集进行田块二值分割、边界监督和距离图监督。
- 提供 `others/download_ftw.py`，用于下载 FTW 国家数据包并转换为项目训练目录。
- 使用 `data/DInterface` 统一创建 train、val、test DataLoader。
- 使用 `model/MInterface` 统一管理模型、损失函数、指标、优化器和学习率调度器。
- 默认模型为 `model/hbg_net.py` 中的 `HbgNet`，主干包含 PVT-v2 风格编码器、边界引导解码器和多任务输出头。
- 保留 `ExampleData` 和 `ExampleNet`，用于不依赖真实 FTW 数据的快速 smoke test。

## 目录结构

```text
.
|-- main.py                         # Lightning 训练入口和命令行参数
|-- pyproject.toml                  # Python 依赖配置
|-- uv.lock                         # uv 锁文件
|-- data/
|   |-- data_interface.py           # LightningDataModule 统一入口
|   |-- ftw_dataset.py              # FTW 数据集读取逻辑
|   `-- example_data.py             # 快速测试用示例数据集
|-- others/
|   `-- download_ftw.py             # FTW 下载、解压和预处理脚本
|-- losses/
|   `-- loss.py                     # HBGNet 多任务损失函数
|-- model/
|   |-- model_interface.py          # LightningModule 统一入口
|   |-- hbg_net.py                  # HBGNet 模型
|   `-- example_net.py              # 快速测试用示例模型
|-- tests/                          # 单元测试
`-- docs/                           # 设计文档和项目图说明
```

本地数据生成后建议放在：

```text
ftw_data/
|-- ftw_origin_data/                # 原始下载和解压后的 FTW 数据
`-- ftw_dataset/                    # 转换后的训练数据
    |-- kenya/
    |   |-- train/
    |   |-- val/
    |   `-- test/
    `-- rwanda/
        |-- train/
        |-- val/
        `-- test/
```

每个 split 下应包含：

```text
image/      # RGB 影像
mask/       # 二值田块掩膜
boundary/   # 边界监督图
dist/       # 距离图监督
```

## 环境部署

推荐使用 `uv` 在项目内创建和管理 Python 环境。项目要求 Python 3.12 或更高版本。

### 1. 安装 uv

如果本机还没有 `uv`，在 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装后重新打开 PowerShell，确认可用：

```powershell
uv --version
```

### 2. 创建项目虚拟环境

在项目根目录执行：

```powershell
uv venv --python 3.12
```

如果已经存在 `.venv`，可以直接复用。后续命令都建议在项目根目录运行。

### 3. 安装依赖

安装运行依赖和测试依赖：

```powershell
uv sync --extra dev
```

如果需要手动安装当前项目，也可以使用：

```powershell
uv pip install -e ".[dev]"
```

### 4. 验证环境

```powershell
uv run python -m pytest -q -p no:cacheprovider
```

`-p no:cacheprovider` 用于避免生成 `.pytest_cache/`。

## 数据下载与预处理

FTW 数据来自 Source Cooperative：

```text
https://data.source.coop/kerner-lab/fields-of-the-world-archive
```

脚本可下载全部 FTW 国家数据，也可只下载指定国家。下载和预处理脚本位于 `others/download_ftw.py`。

### 1. 默认下载和转换

在项目根目录执行：

```powershell
uv run python others\download_ftw.py
```

脚本会执行以下步骤：

1. 下载 `checksum.md5`。
2. 下载配置中指定国家的 zip 包。
3. 校验 MD5。
4. 解压到 `ftw_data/ftw_origin_data/ftw/<country>/`。
5. 读取 FTW 原始影像和标签。
6. 导出训练需要的 `image`、`mask`、`boundary`、`dist` 四类 GeoTIFF。

默认输出目录为：

```text
ftw_data/ftw_dataset
```

### 2. 下载全部国家或指定国家

下载全部国家：

```powershell
uv run python others\download_ftw.py --countries all
```

只下载并解压全部国家，不执行格式转换：

```powershell
uv run python others\download_ftw.py --countries all --download-only
```

下载指定国家：

```powershell
uv run python others\download_ftw.py --countries kenya,rwanda
uv run python others\download_ftw.py --countries kenya --download-only
```

`--countries all` 会根据 Source Cooperative 提供的 `checksum.md5` 自动展开全部可用国家；如果国家名拼写错误，脚本会报错并列出可用国家。

### 3. 通过配置修改下载参数

当前脚本通过 `FTW_CONFIG_DEFAULTS` 控制默认下载参数。需要调整国家、split 或输出目录时，也可以编辑 `others/download_ftw.py` 顶部配置：

```python
FTW_CONFIG_DEFAULTS = {
    "download_root": Path("ftw_data") / "ftw_origin_data",
    "ftw_root": None,
    "output_root": Path("ftw_data") / "ftw_dataset",
    "countries": ["rwanda", "kenya"],
    "splits": "train,val,test",
    "max_samples_per_split": None,
    "image_window": "window_b",
    "reflectance_max": 3000.0,
    "boundary_kernel_size": 3,
    "clean_download": False,
    "download_only": False,
}
```

常用调整：

- 下载全部国家：`"countries": "all"`
- 只下载肯尼亚：`"countries": ["kenya"]`
- 只下载卢旺达和肯尼亚：`"countries": ["rwanda", "kenya"]`
- 只下载并解压，不做格式转换：`"download_only": True`
- 调试时限制每个 split 样本数：`"max_samples_per_split": 20`
- 重新下载并清理旧目录：`"clean_download": True`

### 4. 使用已有原始 FTW 数据

如果已经手动下载并解压了 FTW 原始数据，可以设置 `ftw_root`，跳过下载阶段，仅做格式转换：

```python
FTW_CONFIG_DEFAULTS = {
    "ftw_root": Path("path/to/extracted/ftw"),
    "output_root": Path("ftw_data") / "ftw_dataset",
    "countries": ["kenya"],
    "splits": "train,val,test",
    ...
}
```

也可以直接使用命令行参数：

```powershell
uv run python others\download_ftw.py --ftw-root ftw_data\ftw_origin_data\ftw --countries all
```

`ftw_root` 下需要能找到：

```text
<country>/chips_<country>.parquet
<country>/s2_images/window_b/*.tif
<country>/label_masks/semantic_2class/*.tif
```

### 5. 数据目录校验

转换完成后，可以检查目录是否存在：

```powershell
Get-ChildItem ftw_data\ftw_dataset\kenya\train
```

至少应看到：

```text
image
mask
boundary
dist
```

## 快速运行

### 1. 不依赖真实数据的测试

```powershell
uv run python -m pytest -q -p no:cacheprovider
```

测试会在临时目录构造小型 TIFF 数据，不会读取 `ftw_data/`。

### 2. 使用示例数据做训练链路检查

如果只想验证 Lightning、模型接口和日志流程，可以使用 `ExampleData` 和 `ExampleNet`：

```powershell
uv run python main.py `
  --train_dataset example_data `
  --val_datasets example_data `
  --test_datasets example_data `
  --model_name example_net `
  --loss cross_entropy `
  --metric accuracy `
  --fast_dev_run true `
  --accelerator cpu `
  --devices 1 `
  --num_workers 0
```

### 3. 使用 FTW 数据做快速检查

确认 `ftw_data/ftw_dataset` 已经生成后执行：

```powershell
uv run python main.py `
  --fast_dev_run true `
  --accelerator cpu `
  --devices 1 `
  --num_workers 0 `
  --data_root ftw_data/ftw_dataset `
  --country kenya
```

`fast_dev_run=true` 只跑极少量 batch，适合检查数据、模型、损失函数和 checkpoint 配置是否能串起来。

## 正式训练

单国家训练示例：

```powershell
uv run python main.py `
  --data_root ftw_data/ftw_dataset `
  --country kenya `
  --max_epochs 50 `
  --batch_size 4 `
  --accelerator gpu `
  --devices 1 `
  --num_workers 4 `
  --precision 16-mixed
```

多国家合并训练示例：

```powershell
uv run python main.py `
  --data_root ftw_data/ftw_dataset `
  --country kenya rwanda `
  --max_epochs 50 `
  --batch_size 4 `
  --accelerator gpu `
  --devices 1 `
  --num_workers 4
```

使用所有已下载国家训练：

```powershell
uv run python main.py `
  --data_root ftw_data/ftw_dataset `
  --country all `
  --max_epochs 50 `
  --batch_size 4 `
  --accelerator gpu `
  --devices 1 `
  --num_workers 4
```

`--country all` 会扫描 `ftw_data/ftw_dataset` 下所有包含当前 split 数据的国家目录，并合并为一个训练集。

训练日志默认写入：

```text
logs/hbg_net_ftw
```

checkpoint 由 Lightning 的 `ModelCheckpoint` callback 管理，默认保存最优模型和 `last.ckpt`。`logs/`、`checkpoints/` 和 `*.ckpt` 都不会提交到 Git。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--train_dataset` | `ftw_dataset` | 训练数据集模块名 |
| `--val_datasets` | `ftw_dataset` | 验证数据集模块名，可传多个 |
| `--test_datasets` | `ftw_dataset` | 测试数据集模块名，可传多个 |
| `--data_root` | `ftw_data/ftw_dataset` | 转换后 FTW 数据根目录 |
| `--country` | `kenya` | 训练国家，可传 `all` 或 `kenya rwanda` |
| `--model_name` | `hbg_net` | 模型模块名 |
| `--loss` | `loss_f` | HBGNet 多任务损失 |
| `--metric` | `none` | 指标，支持 `none`、`accuracy`、`recall` |
| `--optimizer` | `adamw` | 优化器，支持 `sgd`、`adam`、`adamw` |
| `--lr_scheduler` | `none` | 学习率调度器，支持 `none`、`step`、`cosine` |
| `--img_size` | `512` | HBGNet 输入尺寸参数 |
| `--return_aux_outputs` | `true` | 是否返回 mask、edge、distance 三个输出 |

## 扩展模型

在 `model/` 下新增 `snake_case.py` 文件，并定义同名 CamelCase 类。例如：

```python
from torch import nn


class MyNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.net = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.net(x)
```

保存为：

```text
model/my_net.py
```

运行时指定：

```powershell
uv run python main.py --model_name my_net
```

`MInterface` 会根据模型构造函数签名自动筛选并传入命令行参数。

## 扩展数据集

在 `data/` 下新增 `snake_case.py` 文件，并定义同名 CamelCase 类。例如：

```python
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train"):
        self.data_root = data_root
        self.split = split

    def __len__(self):
        return 100

    def __getitem__(self, index):
        raise NotImplementedError
```

保存为：

```text
data/my_dataset.py
```

运行时指定：

```powershell
uv run python main.py `
  --train_dataset my_dataset `
  --val_datasets my_dataset `
  --test_datasets my_dataset `
  --data_root path\to\data
```

`DInterface` 会自动为训练、验证和测试阶段传入 `split="train"`、`split="val"`、`split="test"`。

## Git 与数据管理约定

- 不提交 `ftw_data/`。
- 不提交 `.pytest_cache/`、`__pycache__/`、`.venv/`。
- 不提交日志、checkpoint、IDE 配置。
- 大型数据应通过下载脚本或外部数据盘管理。
- 如果需要复现实验，应记录代码提交、数据国家、split、训练参数和 checkpoint 路径。

推送前建议检查：

```powershell
git status --short
git ls-files ftw_data
```

第二条命令应无输出，表示 `ftw_data/` 没有进入 Git 索引。
