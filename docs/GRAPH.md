# Project Graph

Generated for `E:\ZuoProject\ZuoPro`.

## Runtime Architecture

```mermaid
flowchart LR
    CLI["CLI args"] --> Main["main.py"]
    Main --> Parser["build_parser"]
    Main --> DataModule["data.DInterface"]
    Main --> ModelModule["model.MInterface"]
    Main --> Trainer["Lightning Trainer"]
    Main --> Callbacks["ModelSummary / ModelCheckpoint / LR Monitor"]
    Main --> Logger["TensorBoardLogger"]

    DataModule --> DynamicDataset["Dynamic dataset import"]
    DynamicDataset --> ExampleData["data.example_data.ExampleData"]
    DynamicDataset --> FtwDataset["data.ftw_dataset.FtwDataset"]

    FtwDataset --> FtwRoot["ftw_data/ftw_dataset"]
    FtwRoot --> Country["country: kenya / rwanda"]
    Country --> Split["split: train / val / test"]
    Split --> Image["image/*.tif"]
    Split --> Mask["mask/*.tif"]
    Split --> Boundary["boundary/*.tif"]
    Split --> Distance["dist/*.tif"]

    ModelModule --> DynamicModel["Dynamic model import"]
    DynamicModel --> ExampleNet["model.example_net.ExampleNet"]
    DynamicModel --> Field["model.hbgnet.Field"]
    Field --> PVT["model.pvtv2.pvt_v2_b2"]
    Field --> Edge["Laplace edge branch"]
    Field --> Context["near_and_long context fusion"]
    Field --> BoundaryGuide["Boundary-guided modules"]
    Field --> MultiScale["Multi-scale fusion"]
    Field --> Outputs["mask / edge / distance outputs"]

    Trainer --> Fit["trainer.fit"]
    Fit --> DataModule
    Fit --> ModelModule
```

## Source Module Dependencies

```mermaid
flowchart TD
    main["main.py"]
    data_init["data/__init__.py"]
    data_interface["data/data_interface.py"]
    example_data["data/example_data.py"]
    ftw_dataset["data/ftw_dataset.py"]
    model_init["model/__init__.py"]
    model_interface["model/model_interface.py"]
    example_net["model/example_net.py"]
    hbgnet["model/hbgnet.py"]
    pvtv2["model/pvtv2.py"]

    test_main["tests/test_main.py"]
    test_data["tests/test_data_interface.py"]
    test_model["tests/test_model_interface.py"]
    show_tif["tests/show_tif.py"]

    main --> data_init
    main --> model_init
    main --> lightning["lightning"]
    main --> callbacks["lightning.pytorch.callbacks"]
    main --> logger["lightning.pytorch.loggers"]

    data_init --> data_interface
    data_interface --> lightning
    data_interface --> dataloader["torch.utils.data.DataLoader"]
    data_interface --> importlib["importlib dynamic loading"]
    data_interface -. "loads by CLI name" .-> example_data
    data_interface -. "loads by CLI name" .-> ftw_dataset

    ftw_dataset --> torch["torch"]
    ftw_dataset --> torchvision["torchvision.transforms"]
    ftw_dataset --> pil["PIL.Image"]
    ftw_dataset --> numpy["numpy"]
    ftw_dataset --> matplotlib["matplotlib"]

    model_init --> model_interface
    model_interface --> lightning
    model_interface --> torch
    model_interface --> metrics["torchmetrics"]
    model_interface --> optimizers["torch.optim"]
    model_interface --> importlib
    model_interface -. "loads by CLI name" .-> example_net
    model_interface -. "loads by CLI name" .-> hbgnet

    example_net --> torch
    hbgnet --> torch
    hbgnet --> pvtv2
    pvtv2 --> timm["timm"]
    pvtv2 --> torch

    test_main --> main
    test_data --> data_interface
    test_data --> ftw_dataset
    test_model --> model_interface
    show_tif --> rasterio["rasterio"]
    show_tif --> skimage["skimage"]
```

## Key Extension Points

- Add a dataset as `data/<snake_case>.py` with matching `CamelCase` class; select it with `--train_dataset`, `--val_datasets`, or `--test_datasets`.
- Add a model as `model/<snake_case>.py` with matching `CamelCase` class; select it with `--model_name`.
- Dataset and model constructors receive only matching CLI keyword arguments, filtered through `inspect.signature`.
- FTW data is discovered from `ftw_data/ftw_dataset/<country>/<split>/image/*.tif` and paired with sibling `mask`, `boundary`, and `dist` folders.
