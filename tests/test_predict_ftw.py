"""Prediction visualization helper tests."""

import numpy as np
import pytest
import torch


def test_make_visualization_combines_prediction_and_target_rows():
    from others.predict_ftw import make_visualization

    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = np.ones((8, 10), dtype=np.uint8) * 255
    boundary = np.zeros((8, 10), dtype=np.uint8)
    distance = np.full((8, 10), 128, dtype=np.uint8)

    image = make_visualization(
        rgb,
        mask,
        boundary,
        distance,
        mask,
        boundary,
        distance,
    )

    assert image.size == (40, 72)


def test_boundary_falls_back_to_mask_edges():
    from others.predict_ftw import boundary_from_output_or_mask

    mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    mask[:, :, 1:4, 1:4] = True

    boundary = boundary_from_output_or_mask(mask, mask)

    assert boundary.shape == mask.shape
    assert boundary.any()
    assert not boundary[:, :, 2, 2].item()


def test_continuous_tensor_to_image_normalizes_prediction_values():
    from others.predict_ftw import continuous_tensor_to_image

    value = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

    image = continuous_tensor_to_image(value, normalize=True)

    assert image.dtype == np.uint8
    assert image.min() == 0
    assert image.max() == 255


def test_normalize_config_requires_checkpoint():
    from others.predict_ftw import normalize_config

    with pytest.raises(ValueError, match="checkpoint"):
        normalize_config({"checkpoint": None})


def test_normalize_config_accepts_dict_overrides(tmp_path):
    from others.predict_ftw import normalize_config

    config = normalize_config(
        {
            "checkpoint": tmp_path / "model.ckpt",
            "output_dir": tmp_path / "predictions",
            "country": ["kenya"],
        }
    )

    assert config["checkpoint"] == tmp_path / "model.ckpt"
    assert config["output_dir"] == tmp_path / "predictions"
    assert config["country"] == ["kenya"]
