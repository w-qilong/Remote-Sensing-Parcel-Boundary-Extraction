def test_parser_defaults_are_cpu_safe():
    from main import build_parser

    args = build_parser().parse_args([])

    assert args.accelerator == "auto"
    assert args.devices == "auto"
    assert args.precision == "32-true"
    assert args.fast_dev_run is False


def test_parser_exposes_ftw_defaults():
    from main import build_parser

    args = build_parser().parse_args([])

    assert args.data_root == "ftw_data/ftw_dataset"
    assert args.country == ["kenya"]
