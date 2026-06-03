from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _write_tif(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 3:
        image = Image.fromarray(np.moveaxis(array, 0, -1))
    else:
        image = Image.fromarray(array)
    image.save(path)


def _build_ftw_split(root: Path, country: str, split: str, sample_name: str = "sample_0") -> None:
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


def test_data_interface_uses_stage_split(tmp_path):
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
