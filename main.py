"""训练入口。

这个文件是用户启动实验时最常接触的地方，建议先从这里读起。

main.py 只负责三件事：
1. 定义并解析命令行参数；
2. 根据参数创建数据接口 DInterface 和模型接口 MInterface；
3. 创建 Lightning Trainer 并启动训练。

数据集和模型的具体实现不要写在这里，分别放到 data/ 与 model/ 中。
这样做的好处是：换数据集或换模型时，通常只需要新增一个文件，再通过
--train_dataset / --model_name 指定名称即可。
"""

from __future__ import annotations

from argparse import ArgumentParser

import lightning as L
from data import DInterface
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from model import MInterface


def str_to_bool(value: str | bool) -> bool:
    """把命令行中的字符串布尔值转换为 Python bool。

    argparse 默认不会把 "true" / "false" 自动识别成布尔值。本函数允许用户写：
    --fast_dev_run true
    --enable_progress_bar yes
    """
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def build_parser() -> ArgumentParser:
    """集中声明模板支持的所有命令行参数。

    新增实验参数时，优先在这里添加。随后只要 DInterface、MInterface、Dataset
    或 nn.Module 的构造函数声明了同名参数，本模板会自动把对应值传进去。
    """
    parser = ArgumentParser(description="Clean PyTorch Lightning template")

    # 训练控制参数：这些参数直接传给 Lightning Trainer 或影响全局训练行为。
    # 默认值保持保守，保证没有 GPU 的机器也能用 CPU 跑通模板。
    parser.add_argument("--seed", default=1234, type=int)
    # accelerator="auto" 会让 Lightning 自动选择 cpu/gpu/mps 等可用设备。
    parser.add_argument("--accelerator", default="auto")
    # devices="auto" 会使用当前 accelerator 下的默认设备数量；调试时可设为 1。
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--early_stopping_patience", default=3, type=int)
    # 32-true 是最稳妥的全精度训练；需要混合精度时可改成 16-mixed 或 bf16-mixed。
    parser.add_argument("--precision", default="32-true")
    # fast_dev_run 会只跑极少 batch，适合检查数据、模型、loss、日志是否能串起来。
    parser.add_argument("--fast_dev_run", nargs="?", const=True, default=False, type=str_to_bool) # TDOD: 调试时设置为 true，正式训练时设置为 false。
    # TensorBoard 日志会写入 log_dir/experiment_name。
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--experiment_name", default="dino_dpt_ftw")
    parser.add_argument("--enable_progress_bar", nargs="?", const=True, default=True, type=str_to_bool)

    # 数据参数：默认使用 FTW 的 train/val/test split。
    parser.add_argument("--train_dataset", default="ftw_dataset")
    parser.add_argument("--val_datasets", nargs="+", default=["ftw_dataset"])
    parser.add_argument("--test_datasets", nargs="+", default=["ftw_dataset"])
    parser.add_argument("--batch_size", default=16, type=int)
    # Windows 或快速调试时建议先用 0；Linux 训练大数据集时可逐步调高。
    parser.add_argument("--num_workers", default=8, type=int)
    # FTW 数据集默认根目录与国家参数。多个国家时会自动合并到同一个数据集里。
    parser.add_argument("--data_root", default="ftw_data/ftw_dataset")
    parser.add_argument("--country", nargs="+", default=["all"]) # 设置为all时会自动选择所有国家的数据。也可以使用 --country ['<country1>', '<country2>']来指定多个国家。
    # 以下参数由示例数据集 ExampleData 使用；换成真实数据集后可替换为 data_dir 等参数。
    parser.add_argument("--num_samples", default=128, type=int)
    parser.add_argument("--image_size", default=28, type=int)

    # 模型参数：默认加载 model/hbg_net.py 中的 HbgNet。
    parser.add_argument("--model_name", default="dino_dpt")
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--drop_rate", default=0.4, type=float)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--return_aux_outputs", nargs="?", const=True, default=True, type=str_to_bool)
    parser.add_argument("--dino_model_name", default="vit_large_patch16_dinov3.sat493m")
    parser.add_argument("--dino_pretrained", nargs="?", const=True, default=True, type=str_to_bool)
    parser.add_argument("--dino_out_indices", nargs="+", default=None, type=int)
    parser.add_argument("--freeze_backbone", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--trainable_backbone_blocks", default=2, type=int)

    # 优化器、损失和指标参数：由 MInterface 统一解释并创建对应对象。
    # 如果需要新增 loss/metric/optimizer，请同步扩展 model/model_interface.py。
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "mse", "triplet_margin_loss", "bce_with_logits", "loss_f"],
        default="loss_f",
    )
    parser.add_argument("--metric", choices=["accuracy", "recall", "none"], default="none")
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--lr_scheduler", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--lr_decay_steps", default=10, type=int)
    parser.add_argument("--lr_decay_rate", default=0.5, type=float)
    parser.add_argument("--lr_decay_min_lr", default=1e-5, type=float)
    return parser


def build_callbacks(args):
    """创建训练过程中使用的 Lightning callbacks。

    模板默认包含：
    - ModelSummary：启动时打印模型结构，便于确认模型是否正确实例化；
    - ModelCheckpoint：按验证指标保存最优模型和 last checkpoint；
    - LearningRateMonitor：仅在启用学习率调度器时记录学习率变化。
    """
    val_loss_monitor = (
        "val_loss/dataloader_idx_0"
        if len(args.val_datasets) > 1
        else "val_loss"
    )
    metric_name = "val_loss" if args.metric == "none" else f"val_{args.metric}"
    monitor = (
        f"{metric_name}/dataloader_idx_0"
        if len(args.val_datasets) > 1
        else metric_name
    )
    callbacks = [
        ModelSummary(max_depth=2),
        EarlyStopping(
            monitor=val_loss_monitor,
            mode="min",
            patience=args.early_stopping_patience,
        ),
        ModelCheckpoint(
            monitor=monitor,
            mode="min" if args.metric == "none" else "max",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}",
            auto_insert_metric_name=False,
        ),
    ]
    if args.enable_progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=1))
    if args.lr_scheduler != "none":
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def build_trainer(args) -> L.Trainer:
    """根据命令行参数创建 Lightning Trainer。

    Trainer 是 Lightning 的训练调度器，负责设备选择、epoch/batch 循环、日志、
    callback 调用、checkpoint 保存等流程。业务模型逻辑不应该写在这里。
    """
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)
    return L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        logger=logger,
        callbacks=build_callbacks(args),
        log_every_n_steps=1,
        enable_progress_bar=args.enable_progress_bar,
    )


def main(args=None) -> None:
    """组装完整训练流程。

    args=None 时读取真实命令行参数；测试代码也可以传入列表，例如：
    main(["--fast_dev_run", "true", "--accelerator", "cpu"])
    """
    parser = build_parser()
    args = parser.parse_args(args=args)
    # 固定随机种子，workers=True 会同步 DataLoader worker 的随机状态。
    L.seed_everything(args.seed, workers=True)

    # vars(args) 会把 argparse Namespace 转成 dict，方便 DInterface/MInterface
    # 自动筛选自己需要的参数。
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))
    trainer = build_trainer(args)
    # fit 会按 Lightning 标准流程调用 data_module.setup、train_dataloader、
    # model.training_step、model.validation_step 等 hook。
    trainer.fit(model, datamodule=data_module)
    # 训练结束后使用验证集上最优 checkpoint 在测试集上评估模型。
    trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
