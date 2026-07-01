import numpy as np

from data.fbis22m_dataset import make_distance_map


def test_distance_map_is_computed_per_instance_for_touching_masks() -> None:
    """相邻实例应各自计算 distance，不能先合并成一个大连通区域。"""

    instance_masks = np.zeros((2, 7, 8), dtype=np.uint8)
    # 两个矩形实例左右贴边：如果先做 union，它们会变成一个宽矩形，
    # 共享边附近会被错误地视作大区域中心。
    instance_masks[0, 1:6, 1:4] = 1
    instance_masks[1, 1:6, 4:7] = 1

    distance_map = make_distance_map(instance_masks)

    # 每个实例的内部中心被归一化到 1；贴边处仍是各自实例的边界附近，
    # 因而距离值应明显低于实例中心。
    assert distance_map[3, 2] == 1.0
    assert distance_map[3, 5] == 1.0
    assert distance_map[3, 3] < distance_map[3, 2]
    assert distance_map[3, 4] < distance_map[3, 5]


def test_distance_map_handles_empty_instance_masks() -> None:
    """空实例输入仍应返回稳定的二维 float32 distance target。"""

    distance_map = make_distance_map(np.zeros((0, 7, 8), dtype=np.uint8))

    assert distance_map.shape == (7, 8)
    assert distance_map.dtype == np.float32
    assert distance_map.max() == 0.0
