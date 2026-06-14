"""训练入口参数的回归测试。

这些测试只检查 argparse 默认值，不启动 Lightning Trainer，也不读取真实 FTW 数据。
这样可以快速发现默认训练配置被意外改回示例模型或不适合本项目数据的情况。
"""


def test_parser_defaults_are_cpu_safe():
    """默认 Trainer 参数应能在无 GPU 的机器上安全解析。"""
    from main import build_parser

    args = build_parser().parse_args([])

    assert args.accelerator == "auto"
    assert args.devices == "auto"
    assert args.precision == "32-true"
    assert args.fast_dev_run is False


def test_parser_exposes_ftw_defaults():
    """默认业务配置应指向 FTW 数据集、HBGNet 模型和多任务损失。"""
    from main import build_parser

    args = build_parser().parse_args([])

    assert args.train_dataset == "ftw_dataset"
    assert args.val_datasets == ["ftw_dataset"]
    assert args.test_datasets == ["ftw_dataset"]
    assert args.data_root == "ftw_data/ftw_dataset"
    assert args.country == ["kenya"]
    assert args.model_name == "hbg_net"
    assert args.in_channels == 3
    assert args.num_classes == 2
    assert args.loss == "loss_f"
    assert args.metric == "none"
    assert args.return_aux_outputs is True


def test_parser_accepts_all_and_arbitrary_ftw_countries():
    """训练入口不应把 FTW 国家限制在 kenya/rwanda。"""
    from main import build_parser

    parser = build_parser()

    assert parser.parse_args(["--country", "all"]).country == ["all"]
    assert parser.parse_args(["--country", "germany", "france"]).country == [
        "germany",
        "france",
    ]


def test_checkpoint_monitor_matches_number_of_validation_loaders():
    """单验证集不带 dataloader 后缀，多验证集才监控第一个验证集。"""
    from lightning.pytorch.callbacks import ModelCheckpoint
    from main import build_callbacks, build_parser

    parser = build_parser()

    single_args = parser.parse_args([])
    single_checkpoint = next(
        callback
        for callback in build_callbacks(single_args)
        if isinstance(callback, ModelCheckpoint)
    )
    assert single_checkpoint.monitor == "val_loss"

    multi_args = parser.parse_args(["--val_datasets", "ftw_dataset", "example_data"])
    multi_checkpoint = next(
        callback
        for callback in build_callbacks(multi_args)
        if isinstance(callback, ModelCheckpoint)
    )
    assert multi_checkpoint.monitor == "val_loss/dataloader_idx_0"


def test_lightning_progress_bar_is_enabled_by_default():
    """默认使用 Lightning 进度条显示 epoch 和 step 进度。"""
    from lightning.pytorch.callbacks import TQDMProgressBar
    from main import build_callbacks, build_parser

    parser = build_parser()
    args = parser.parse_args([])

    assert any(
        isinstance(callback, TQDMProgressBar)
        for callback in build_callbacks(args)
    )

    disabled_args = parser.parse_args(["--enable_progress_bar", "false"])
    assert not any(
        isinstance(callback, TQDMProgressBar)
        for callback in build_callbacks(disabled_args)
    )


def test_early_stopping_monitors_val_loss_with_patience_three():
    """EarlyStopping 应在 val_loss 连续 3 个 epoch 不下降时停止训练。"""
    from lightning.pytorch.callbacks import EarlyStopping
    from main import build_callbacks, build_parser

    parser = build_parser()

    single_args = parser.parse_args([])
    single_early_stopping = next(
        callback
        for callback in build_callbacks(single_args)
        if isinstance(callback, EarlyStopping)
    )
    assert single_early_stopping.monitor == "val_loss"
    assert single_early_stopping.patience == 3
    assert single_early_stopping.mode == "min"

    multi_args = parser.parse_args(["--val_datasets", "ftw_dataset", "example_data"])
    multi_early_stopping = next(
        callback
        for callback in build_callbacks(multi_args)
        if isinstance(callback, EarlyStopping)
    )
    assert multi_early_stopping.monitor == "val_loss/dataloader_idx_0"


def test_main_tests_best_checkpoint_after_fit(monkeypatch):
    """main 训练结束后应使用 best checkpoint 进行测试。"""
    import main as train_main

    calls = []

    class FakeTrainer:
        def fit(self, model, datamodule):
            calls.append(("fit", model, datamodule))

        def test(self, model, datamodule, ckpt_path):
            calls.append(("test", model, datamodule, ckpt_path))

    monkeypatch.setattr(train_main.L, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_main, "DInterface", lambda **kwargs: "data_module")
    monkeypatch.setattr(train_main, "MInterface", lambda **kwargs: "model")
    monkeypatch.setattr(train_main, "build_trainer", lambda args: FakeTrainer())

    train_main.main(["--fast_dev_run", "true", "--enable_progress_bar", "false"])

    assert calls == [
        ("fit", "model", "data_module"),
        ("test", "model", "data_module", "best"),
    ]
