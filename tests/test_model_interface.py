"""模型接口的训练路径回归测试。

这里不直接实例化完整 HBGNet 大模型；普通分类路径使用 ``ExampleNet``，多任务路径用
极小的 ``TinyHbgNet`` 替身验证 ``LossF`` batch 解析和损失计算，保证测试快速稳定。
"""

import torch
from torch import nn


def _batch(batch_size=2):
    """生成普通分类任务使用的最小 batch。"""
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


def test_model_interface_runs_training_step_and_optimizer():
    """MInterface 应能跑通普通分类训练 step，并创建指定优化器。"""
    from model import MInterface

    module = MInterface(
        model_name="example_net",
        loss="cross_entropy",
        metric="accuracy",
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        num_classes=10,
        in_channels=1,
    )

    loss = module.training_step(_batch(), 0)
    optimizer = module.configure_optimizers()

    assert loss.ndim == 0
    assert loss.requires_grad
    assert isinstance(optimizer, torch.optim.Adam)


def test_model_interface_runs_loss_f_training_step(monkeypatch):
    """MInterface 应能用 LossF 跑通 FTW/HBGNet 风格的多任务 batch。"""
    from model import MInterface

    class TinyHbgNet(nn.Module):
        """用于测试的极小三输出网络，避免实例化完整 PVT 主干。"""

        def __init__(self):
            super().__init__()
            self.mask_head = nn.Conv2d(3, 1, kernel_size=1)
            self.edge_head = nn.Conv2d(3, 2, kernel_size=1)
            self.dist_head = nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, images):
            return [
                self.mask_head(images),
                self.edge_head(images).log_softmax(dim=1),
                self.dist_head(images),
            ]

    monkeypatch.setattr(MInterface, "_load_model", lambda self, model_name, extra_kwargs: TinyHbgNet())

    module = MInterface(
        model_name="hbg_net",
        loss="loss_f",
        metric="none",
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        num_classes=2,
        in_channels=3,
    )
    images = torch.randn(2, 3, 8, 8)
    masks = torch.randint(0, 2, (2, 1, 8, 8)).float()
    contours = torch.randint(0, 2, (2, 1, 8, 8)).long()
    distances = torch.rand(2, 1, 8, 8)

    loss = module.training_step((["a", "b"], images, masks, contours, distances), 0)

    assert loss.ndim == 0
    assert loss.requires_grad
