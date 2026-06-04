"""FTW 数据集加载工具。

这个模块把旧的 HBGNet 风格 GeoTIFF 读取逻辑，改造成当前项目模板可直接使用的
数据集实现。核心约定很简单：数据集会从 ``ftw_data/ftw_dataset/<country>/<split>/image``
中自动发现样本，并从同级的 ``mask``、``boundary`` 和 ``dist`` 目录读取对应标签。

为了便于后续维护，这个文件尽量把“路径怎么拼”“标签怎么读”“多个国家怎么合并”
都写在同一个地方，避免在训练入口和数据接口里重复处理同样的规则。
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_tif(file_name, xoff=0, yoff=0, data_width=0, data_height=0):
    """读取 TIFF 文件，并返回基础元数据和像素数据。

    这里保留了旧接口参数 ``xoff``、``yoff``、``data_width``、``data_height``，
    主要是为了兼容原始 HBGNet 代码的调用方式。当前项目里暂时没有使用窗口读取，
    所以这些参数只是占位保留。
    """
    # 当前实现直接整幅读取，后续如果需要做大图窗口切片，可以在这里扩展。
    image = Image.open(file_name)
    data = np.array(image)
    # 如果是彩色图像，PIL 读取后通常是 HWC，这里转成 CHW 方便后续统一处理。
    if data.ndim == 3:
        data = np.moveaxis(data, -1, 0)
    width, height = image.size
    count = 1 if data.ndim == 2 else data.shape[0]
    return width, height, count, data, None, None


def write_tiff(im_data, im_geotrans, im_proj, path):
    """将数组写回 TIFF 文件。

    这个函数保留原名，方便旧代码继续复用。当前版本不保存空间参考信息，
    因为项目里的训练与测试暂时只关心图像内容本身；如果后续需要恢复 GIS 流程，
    可以再把 ``im_geotrans`` 和 ``im_proj`` 接回去。
    """
    # 这两个参数暂时不参与写文件，但保留它们可以保持函数签名兼容。
    del im_geotrans, im_proj
    # Pillow 需要按 HWC 格式保存，所以这里把 CHW 转回去。
    if im_data.ndim == 3:
        image = Image.fromarray(np.moveaxis(im_data, 0, -1))
    else:
        image = Image.fromarray(im_data)
    image.save(path)


def companion_path(image_path: str, target_kind: str) -> str:
    """根据影像路径推导对应标签路径。

    目录结构约定为：
    ``<root>/<country>/<split>/image/<name>.tif`` ->
    ``<root>/<country>/<split>/<target_kind>/<name>.tif``
    """
    # 这里的约定非常重要：训练时影像与标签必须同名，否则就无法自动配对。
    path = Path(image_path)
    image_dir = path.parent.name
    if image_dir != "image":
        raise ValueError(f"Expected image path under an 'image' folder: {image_path}")
    # 先回到 split 目录，再拼出目标标签子目录。
    return str(path.parent.parent / target_kind / path.name)


def _read_single_band(path: str):
    """读取单波段标签；如果意外是多波段，则保留第一波段。

    FTW 的监督标签理论上是单通道，但现实数据里也可能出现保存格式不一致的情况。
    这里采用“只取第一波段”的策略，避免多波段标签直接把训练流程打断。
    """

    _, _, _, data, _, _ = read_tif(path)
    if data.ndim == 3:
        # 多波段时只使用第一波段，和二值标签的训练习惯保持一致。
        data = data[0]
    return data


def load_image(path: str):
    """读取 RGB 影像，并按项目模型主干需要的方式做归一化。

    这里使用 ImageNet 归一化，是为了和现有模型主干的预训练分布对齐。
    如果后续更换主干，这里的均值和方差通常也需要一起调整。
    """

    # 强制转成 RGB，避免灰度图或带透明通道的 TIFF 影响模型输入通道数。
    image = Image.open(path).convert("RGB")

    # ToTensor 会把像素值变成 [0, 1] 的浮点张量，Normalize 再做标准化。
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return data_transforms(image)


def load_mask(path: str):
    # mask 使用 0/255 存储，训练时统一压到 0/1 的浮点张量。
    mask = _read_single_band(companion_path(path, "mask")).astype("uint8") / 255.0
    return torch.from_numpy(np.expand_dims(mask, 0)).float()


def load_contour(path: str):
    # 边界分支通常期望 long 类型标签，这里保持和现有损失函数兼容。
    contour = _read_single_band(companion_path(path, "boundary")).astype("uint8") / 255.0
    return torch.from_numpy(np.expand_dims(contour, 0)).long()


def load_distance(path: str):
    # 距离图同样按 0/1 浮点张量处理，便于后续回归或回归式监督。
    distance = _read_single_band(companion_path(path, "dist")).astype("uint8") / 255.0
    return torch.from_numpy(np.expand_dims(distance, 0)).float()


def _as_list(value: str | Sequence[str]) -> list[str]:
    """把单个国家名或多个国家名统一成列表。

    这样做是为了让命令行既可以传 ``--country kenya``，也可以传
    ``--country kenya rwanda``。后续合并多个国家时，数据集内部只处理列表。
    """

    if isinstance(value, str):
        return [value]
    return list(value)


def _resolve_data_root(data_root: str) -> Path:
    path = Path(data_root)
    if path.is_absolute() or path.exists():
        return path

    project_root = Path(__file__).resolve().parents[1]
    project_relative = project_root / path
    if project_relative.exists():
        return project_relative

    return path


class FtwDataset(Dataset):
    """FTW 训练/验证/测试数据集。

    默认会从当前 split 的 ``image`` 目录扫描样本名；如果传入
    ``file_names``，则优先使用外部指定的样本列表。
    """

    def __init__(
        self,
        data_root: str = "ftw_data/ftw_dataset",
        country: str | Sequence[str] = "kenya",
        split: str = "train",
        file_names: list[str] | None = None,
    ) -> None:
        # 根目录保存为 Path，后续拼接子目录时更清晰，也更不容易写错斜杠。
        self.data_root = _resolve_data_root(data_root)
        # country 参数既支持单个字符串，也支持字符串列表；统一转成列表后再处理。
        self.countries = _as_list(country)
        self.split = split
        # 当 country 多于一个时，我们把多个国家的数据拼接成一个长数据集。
        self.multiple_countries = len(self.countries) > 1
        # samples 里保存的是 (country_name, image_name) 二元组，避免不同国家之间重名冲突。
        self.samples: list[tuple[str, str]] = []

        for country_name in self.countries:
            # 每个国家都有自己独立的 split 目录，例如 kenya/train 或 rwanda/val。
            root_dir = self.data_root / country_name / self.split
            image_dir = root_dir / "image"

            # 目录不存在时直接报错，避免后面在训练中才发现数据路径写错。
            if not image_dir.exists():
                raise FileNotFoundError(f"FTW image directory not found: {image_dir}")

            # file_names 为空时，自动扫描当前国家当前 split 下的所有 tif 文件。
            if file_names is None:
                country_file_names = sorted(path.stem for path in image_dir.glob("*.tif"))
            else:
                # 如果外部显式传入样本名，就直接使用它；这在测试和局部调试时很方便。
                country_file_names = list(file_names)

            # 每个国家的样本按国家顺序依次追加，这样多个国家合并时顺序是可预期的。
            self.samples.extend((country_name, image_name) for image_name in country_file_names)

        # file_names 是一个“展示名”列表，方便调试时查看样本来源。
        self.file_names = [self._display_name(country_name, image_name) for country_name, image_name in self.samples]

    def _display_name(self, country_name: str, image_name: str) -> str:
        # 多国家时加上国家前缀，避免 train/val/test 中不同国家的同名样本混淆。
        if self.multiple_countries:
            return f"{country_name}/{image_name}"
        return image_name

    def __len__(self):
        # 数据集长度就是最终拼接后的样本总数。
        return len(self.samples)

    def _image_path(self, country_name: str, image_name: str) -> str:
        # 根据国家名和样本名拼出真正的影像路径。
        return str(self.data_root / country_name / self.split / "image" / f"{image_name}.tif")

    def __getitem__(self, idx):
        # 从 samples 中取出“属于哪个国家”“样本名是什么”，再回到对应目录读取。
        country_name, image_name = self.samples[idx]
        image_path = self._image_path(country_name, image_name)
        image = load_image(image_path)
        # 训练时需要同时返回 mask、边界和距离图，保持和现有模型接口一致。
        mask = load_mask(image_path)
        contour = load_contour(image_path)
        dist = load_distance(image_path)
        return self._display_name(country_name, image_name), image, mask, contour, dist


class FtwPredictionDataset(FtwDataset):
    """仅用于预测或推理的 FTW 影像数据集。"""

    def __getitem__(self, idx):
        # 预测阶段只需要影像和样本名，不再读取标签。
        country_name, image_name = self.samples[idx]
        image_path = self._image_path(country_name, image_name)
        return self._display_name(country_name, image_name), load_image(image_path)

if __name__ == "__main__":
    # 这个模块主要提供数据集实现，不直接暴露命令行接口；如果需要测试数据加载，可以在 tests/ 里写专门的测试用例。
    country = ['kenya', 'rwanda']
    dataset = FtwDataset(country=country)
    for name, image, mask, contour, dist in dataset:
        if name == "rwanda/1592589":
            print(f"Sample: name={name}, image_shape={image.shape}, mask_shape={mask.shape}, contour_shape={contour.shape}, dist_shape={dist.shape}")
            # 可视化检查读取的图像和标签是否正确；实际训练时可以注释掉这个部分。
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            # 注意 image 是经过归一化的张量，显示时需要转回 HWC 格式，并且可能需要反归一化才能正确显示颜色。
            # 反归一化的公式是：img = img * std + mean，其中 mean 和 std 是 load_image 中使用的 ImageNet 归一化参数。
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image.permute(1, 2, 0).numpy() * std + mean  # 转回 HWC 并反归一化
            axes[0].imshow(image_np)
            axes[0].set_title("Image")
            axes[1].imshow(mask.squeeze(), cmap="gray")
            axes[1].set_title("Mask")
            axes[2].imshow(contour.squeeze(), cmap="gray")
            axes[2].set_title("Contour")
            axes[3].imshow(dist.squeeze(), cmap="gray")
            axes[3].set_title("Distance")
            plt.show()

        
    
