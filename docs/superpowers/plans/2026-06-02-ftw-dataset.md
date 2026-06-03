# FTW Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the HBGNet-style dataset code into a project-native FTW dataset module that plugs into `DInterface` and works with the existing Lightning template.

**Architecture:** Keep one reusable dataset module that knows how to discover FTW samples from `ftw_data/ftw_dataset/<country>/<split>/image`, then let `DInterface` inject the active split (`train` / `val` / `test`) when it instantiates datasets. This keeps the public template interface simple while preserving one code path for image and label loading.

**Tech Stack:** Python 3.12, PyTorch, torchvision, rasterio, PyTorch Lightning, pytest.

---

### Task 1: Add FTW dataset module

**Files:**
- Create: `data/ftw_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
from data.ftw_dataset import FtwDataset


def test_ftw_dataset_scans_samples(tmp_path):
    dataset = FtwDataset(data_root=str(tmp_path), country="kenya", split="train")
    assert len(dataset) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_interface.py::test_ftw_dataset_scans_samples -v`
Expected: FAIL because `data/ftw_dataset.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class FtwDataset(Dataset):
    def __init__(self, data_root: str = "ftw_data/ftw_dataset", country: str = "kenya", split: str = "train", file_names: list[str] | None = None):
        ...
```

The module must:
- scan `data_root / country / split / image` when `file_names` is `None`
- load `image`, `mask`, `boundary`, and `dist` by matching the same file name
- keep `FtwPredictionDataset` for image-only inference
- keep compatibility aliases for the older `HBGNetDataset` names if they are useful

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_interface.py::test_ftw_dataset_scans_samples -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/ftw_dataset.py
git commit -m "feat: add ftw dataset module"
```

### Task 2: Inject split in DInterface

**Files:**
- Modify: `data/data_interface.py`

- [ ] **Step 1: Write the failing test**

```python
import torch


def test_data_interface_uses_stage_split(tmp_path):
    from data import DInterface

    dm = DInterface(
        train_dataset="ftw_dataset",
        val_datasets=["ftw_dataset"],
        test_datasets=["ftw_dataset"],
        data_root=str(tmp_path / "ftw_dataset"),
        country="kenya",
        batch_size=1,
        num_workers=0,
    )

    dm.setup("fit")
    assert dm.train_set.split == "train"
    assert dm.val_sets[0].split == "val"

    dm.setup("test")
    assert dm.test_sets[0].split == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_interface.py::test_data_interface_uses_stage_split -v`
Expected: FAIL because `DInterface` currently passes the same kwargs to every split.

- [ ] **Step 3: Write minimal implementation**

Update `_instantiate_dataset` to accept override kwargs and pass `split="train"`, `split="val"`, or `split="test"` from `setup()`.

```python
    def _instantiate_dataset(self, dataset_name: str, **override_kwargs):
        dataset_cls = self._load_dataset_class(dataset_name)
        signature = inspect.signature(dataset_cls.__init__)
        accepted = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        merged_kwargs = dict(self.kwargs)
        merged_kwargs.update(override_kwargs)
        dataset_kwargs = {name: merged_kwargs[name] for name in accepted if name in merged_kwargs}
        return dataset_cls(**dataset_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_interface.py::test_data_interface_uses_stage_split -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/data_interface.py
git commit -m "feat: pass stage split to datasets"
```

### Task 3: Update parser and tests for FTW defaults

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`
- Modify: `tests/test_data_interface.py`

- [ ] **Step 1: Write the failing test**

```python
def test_parser_exposes_ftw_defaults():
    from main import build_parser

    args = build_parser().parse_args([])
    assert args.data_root == "ftw_data/ftw_dataset"
    assert args.country == "kenya"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py::test_parser_exposes_ftw_defaults -v`
Expected: FAIL because the arguments do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add command line arguments for `--data_root` and `--country` in `build_parser()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py::test_parser_exposes_ftw_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py tests/test_data_interface.py
git commit -m "feat: wire ftw dataset defaults"
```
