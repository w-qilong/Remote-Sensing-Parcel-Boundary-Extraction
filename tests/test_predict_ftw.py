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


def test_make_visualization_adds_boundary_probability_column():
    from others.predict_ftw import make_visualization

    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = np.ones((8, 10), dtype=np.uint8) * 255
    boundary = np.zeros((8, 10), dtype=np.uint8)
    distance = np.full((8, 10), 128, dtype=np.uint8)
    probability = np.full((8, 10), 64, dtype=np.uint8)

    image = make_visualization(
        rgb,
        mask,
        boundary,
        distance,
        mask,
        boundary,
        distance,
        predicted_boundary_probability=probability,
    )

    assert image.size == (50, 72)


def test_boundary_falls_back_to_mask_edges():
    from others.predict_ftw import boundary_from_output_or_mask

    mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    mask[:, :, 1:4, 1:4] = True

    boundary = boundary_from_output_or_mask(mask, mask)

    assert boundary.shape == mask.shape
    assert boundary.any()
    assert not boundary[:, :, 2, 2].item()


def test_boundary_probability_uses_foreground_channel():
    from others.predict_ftw import boundary_probability_from_output_or_mask

    edge_scores = torch.tensor(
        [
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 2.0], [2.0, 0.0]],
            ]
        ]
    )
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    probability = boundary_probability_from_output_or_mask([mask.float(), edge_scores], mask)

    assert probability.shape == (1, 1, 2, 2)
    assert probability[0, 0, 0, 1] > probability[0, 0, 0, 0]


def test_boundary_threshold_can_keep_non_argmax_boundary_probability():
    from others.predict_ftw import boundary_from_output_or_mask

    edge_scores = torch.tensor([[[[0.0]], [[-0.5]]]])
    mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool)

    boundary = boundary_from_output_or_mask([mask.float(), edge_scores], mask, threshold=0.3)

    assert boundary.item()


def test_postprocess_boundary_closes_small_gaps():
    from others.predict_ftw import postprocess_boundary

    boundary = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    boundary[:, :, 2, 1] = True
    boundary[:, :, 2, 3] = True

    processed = postprocess_boundary(boundary, close_kernel_size=3, close_iterations=1)

    assert processed[:, :, 2, 2].item()


def test_postprocess_boundary_rejects_even_kernel_size():
    from others.predict_ftw import postprocess_boundary

    boundary = torch.zeros(1, 1, 5, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="positive odd"):
        postprocess_boundary(boundary, close_kernel_size=2)


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
