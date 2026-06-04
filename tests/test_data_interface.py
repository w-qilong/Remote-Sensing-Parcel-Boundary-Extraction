"""数据接口和 FTW 数据集的回归测试。

测试会在临时目录中构造极小的 TIFF 样本，避免依赖仓库里的真实 ``ftw_data``。
这样既能验证目录扫描、标签配对和多国家合并逻辑，也不会让测试运行时间受真实数据规模影响。
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _write_tif(path: Path, array: np.ndarray) -> None:
    """把测试用 numpy 数组写成 TIFF，模拟真实 FTW 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 3:
        image = Image.fromarray(np.moveaxis(array, 0, -1))
    else:
        image = Image.fromarray(array)
    image.save(path)


def _build_ftw_split(root: Path, country: str, split: str, sample_name: str = "sample_0") -> None:
    """构造一个最小 FTW split 目录，包含 image/mask/boundary/dist 四类文件。"""
    base = root / country / split
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[0, :, :] = 10
    mask = np.full((4, 4), 255, dtype=np.uint8)
    boundary = np.zeros((4, 4), dtype=np.uint8)
    boundary[1, 1] = 255
    dist = np.full((4, 4), 128, dtype=np.uint8)

    _write_tif(base / "image" / f"{sample_name}.tif", image)
    _write_tif(base / "mask" / f"{sample_name}.tif", mask)
    _write_tif(base / "boundary" / f"{sample_name}.tif", boundary)
    _write_tif(base / "dist" / f"{sample_name}.tif", dist)


def test_ftw_dataset_scans_samples(tmp_path):
    """FtwDataset 应能自动扫描 image 目录并读取配套标签。"""
    from data.ftw_dataset import FtwDataset

    _build_ftw_split(tmp_path / "ftw_dataset", "kenya", "train")

    dataset = FtwDataset(data_root=str(tmp_path / "ftw_dataset"), country="kenya", split="train")

    assert len(dataset) == 1
    sample_name, image, mask, contour, dist = dataset[0]
    assert sample_name == "sample_0"
    assert image.shape == (3, 4, 4)
    assert mask.shape == (1, 4, 4)
    assert contour.shape == (1, 4, 4)
    assert dist.shape == (1, 4, 4)
    assert mask.dtype == torch.float32
    assert contour.dtype == torch.int64
    assert dist.dtype == torch.float32


def test_ftw_dataset_merges_multiple_countries(tmp_path):
    """多个国家合并时，样本展示名应带国家前缀以避免重名。"""
    from data.ftw_dataset import FtwDataset

    _build_ftw_split(tmp_path / "ftw_dataset", "kenya", "train", sample_name="same_name")
    _build_ftw_split(tmp_path / "ftw_dataset", "rwanda", "train", sample_name="same_name")

    dataset = FtwDataset(
        data_root=str(tmp_path / "ftw_dataset"),
        country=["kenya", "rwanda"],
        split="train",
    )

    assert len(dataset) == 2
    first_name, *_ = dataset[0]
    second_name, *_ = dataset[1]
    assert first_name == "kenya/same_name"
    assert second_name == "rwanda/same_name"


def test_ftw_dataset_resolves_relative_root_from_project_root(tmp_path, monkeypatch):
    """FtwDataset should still find project data when launched from another cwd."""
    from data import ftw_dataset
    from data.ftw_dataset import FtwDataset

    project_root = tmp_path / "project"
    fake_module = project_root / "data" / "ftw_dataset.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.touch()
    ftw_root = project_root / "ftw_data" / "ftw_dataset"
    _build_ftw_split(ftw_root, "kenya", "train", sample_name="project_relative")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(ftw_dataset, "__file__", str(fake_module))
    monkeypatch.chdir(elsewhere)

    dataset = FtwDataset(data_root="ftw_data/ftw_dataset", country="kenya", split="train")

    assert dataset.data_root == ftw_root
    assert dataset.file_names == ["project_relative"]


def test_data_interface_uses_stage_split(tmp_path):
    """DInterface 应按 Lightning stage 创建 train/val/test 对应 split。"""
    from data import DInterface

    ftw_root = tmp_path / "ftw_dataset"
    _build_ftw_split(ftw_root, "kenya", "train", sample_name="train_sample")
    _build_ftw_split(ftw_root, "kenya", "val", sample_name="val_sample")
    _build_ftw_split(ftw_root, "kenya", "test", sample_name="test_sample")

    dm = DInterface(
        train_dataset="ftw_dataset",
        val_datasets=["ftw_dataset"],
        test_datasets=["ftw_dataset"],
        data_root=str(ftw_root),
        country=["kenya"],
        batch_size=1,
        num_workers=0,
    )

    dm.setup("fit")
    assert dm.train_set.split == "train"
    assert dm.val_sets[0].split == "val"

    train_name, images, masks, contours, distances = next(iter(dm.train_dataloader()))
    assert tuple(train_name) == ("train_sample",)
    assert images.shape == (1, 3, 4, 4)
    assert masks.shape == (1, 1, 4, 4)
    assert contours.shape == (1, 1, 4, 4)
    assert distances.shape == (1, 1, 4, 4)

    dm.setup("test")
    assert dm.test_sets[0].split == "test"

    test_name, test_images, *_ = next(iter(dm.test_dataloader()[0]))
    assert tuple(test_name) == ("test_sample",)
    assert test_images.shape == (1, 3, 4, 4)


def test_data_interface_merges_multiple_countries(tmp_path):
    """DInterface 传入多个国家时，应把它们合并为同一个训练集。"""
    from data import DInterface

    ftw_root = tmp_path / "ftw_dataset"
    _build_ftw_split(ftw_root, "kenya", "train", sample_name="shared_sample")
    _build_ftw_split(ftw_root, "kenya", "val", sample_name="kenya_val")
    _build_ftw_split(ftw_root, "kenya", "test", sample_name="kenya_test")
    _build_ftw_split(ftw_root, "rwanda", "train", sample_name="shared_sample")
    _build_ftw_split(ftw_root, "rwanda", "val", sample_name="rwanda_val")
    _build_ftw_split(ftw_root, "rwanda", "test", sample_name="rwanda_test")

    dm = DInterface(
        train_dataset="ftw_dataset",
        val_datasets=["ftw_dataset"],
        test_datasets=["ftw_dataset"],
        data_root=str(ftw_root),
        country=["kenya", "rwanda"],
        batch_size=1,
        num_workers=0,
    )

    dm.setup("fit")

    assert len(dm.train_set) == 2
    assert dm.train_set.file_names == ["kenya/shared_sample", "rwanda/shared_sample"]

    first_name, *_ = dm.train_set[0]
    second_name, *_ = dm.train_set[1]
    assert first_name == "kenya/shared_sample"
    assert second_name == "rwanda/shared_sample"


def test_data_interface_builds_fake_data_loaders(tmp_path):
    """模板示例数据集仍应可用，保证项目保留快速 smoke test 能力。"""
    from data import DInterface

    dm = DInterface(
        train_dataset="example_data",
        val_datasets=["example_data"],
        test_datasets=["example_data"],
        data_dir=str(tmp_path),
        batch_size=2,
        num_workers=0,
        image_size=28,
        num_classes=10,
        num_samples=8,
    )

    dm.setup("fit")
    images, labels = next(iter(dm.train_dataloader()))

    assert images.shape == (2, 1, 28, 28)
    assert labels.dtype == torch.long
    assert len(dm.val_dataloader()) == 1

    dm.setup("test")
    assert len(dm.test_dataloader()) == 1
