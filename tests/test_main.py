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
