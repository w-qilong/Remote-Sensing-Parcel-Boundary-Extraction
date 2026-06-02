# FTW 数据集重构设计

## 背景

当前项目的数据模块已经有统一入口 `DInterface`，它会根据 `--train_dataset`、`--val_datasets` 和 `--test_datasets` 自动导入 `data/` 下的 Dataset 类。模板数据集 `ExampleData` 已经符合这个约定，但现有的 HBGNet 数据集代码仍然保留了旧项目风格：类名与文件名不完全匹配模板约定，构造参数也更偏向旧代码，而不是当前项目的命令行式配置。

与此同时，仓库里已经存在 FTW 数据文件结构：`ftw_data/ftw_dataset/kenya` 和 `ftw_data/ftw_dataset/rwanda` 下都有 `train/`、`val/`、`test/`，每个 split 内又分为 `image/`、`mask/`、`boundary/`、`dist/`。这说明数据集代码应该被整理成“按国家 + split 自动定位样本”的项目级 Dataset，而不是继续保留单独的旧工程接口。

## 目标

1. 让 FTW 数据集能被 `DInterface` 直接加载，命名方式遵守 `snake_case.py` + `CamelCase` 类名规则。
2. 用项目模板风格重构数据集入口，让它通过构造参数接收 `data_root`、`country`、`split`、`file_names` 等配置。
3. 保留现有标签读取逻辑：影像、mask、boundary、dist 仍然按同名文件配对读取。
4. 保持训练/验证/测试/预测之间的接口一致，减少后续换数据或扩展 split 时的重复代码。

## 非目标

1. 不改模型结构和训练策略。
2. 不重写 `DInterface` 的整体设计。
3. 不引入新的数据格式转换流程，不改变已有 `.tif` 监督文件内容。
4. 不把数据集代码扩展成独立的清单管理系统或缓存系统。

## 方案

### 1. 数据集模块标准化

将现有数据集逻辑整理为项目内一个新的标准模块，例如 `data/ftw_dataset.py`。模块内提供两个类：

1. `FtwDataset`：训练/验证/测试使用，返回 `sample_name, image, mask, contour, dist`。
2. `FtwPredictionDataset`：推理使用，只返回 `sample_name, image`。

这两个类都遵守模板约定，能被 `DInterface` 按模块名自动加载。

### 2. 统一构造参数

`FtwDataset` 的构造函数只保留项目级参数：

1. `data_root`：FTW 数据根目录，默认指向仓库中的 `ftw_data/ftw_dataset`。
2. `country`：`kenya` 或 `rwanda`。
3. `split`：`train`、`val` 或 `test`。
4. `file_names`：可选的显式样本名列表；如果为空，则自动扫描 `image/` 目录生成。

这样既支持模板式命令行配置，也保留测试中直接传入样本名的能力。

### 3. 目录与样本发现规则

Dataset 在初始化时根据 `data_root / country / split` 定位数据目录。样本名从 `image/*.tif` 自动推导，`__getitem__` 再通过同名文件去读取 `mask/`、`boundary/` 和 `dist/`。

这一层不依赖额外索引文件，避免把当前仓库已有的目录结构再拆成另一套配置格式。

### 4. 标签读取约定

沿用现有读取函数的职责分离：

1. `load_image` 负责 RGB 影像读取与标准化。
2. `load_mask` 负责二值掩膜。
3. `load_contour` 负责边界监督。
4. `load_distance` 负责距离图。

如果某个标签文件缺失或目录结构不符合约定，Dataset 在实例化或取样阶段抛出清晰异常，帮助用户快速定位数据问题。

## 数据流

1. 用户在命令行中指定 `--train_dataset ftw_dataset`，并通过新增参数指定 `--data_root`、`--country`、`--split`。
2. `main.py` 把参数传给 `DInterface`。
3. `DInterface` 根据文件名导入 `data/ftw_dataset.py`，并只把 `FtwDataset.__init__` 接受的参数传进去。
4. `FtwDataset` 自动扫描当前 split 的影像文件名。
5. `DataLoader` 读取 batch，返回训练所需的多监督数据。

预测阶段的流程与训练一致，只是切换到 `FtwPredictionDataset`，从而不再读取标签文件。

## 兼容性

1. 保留 `ExampleData` 不动，继续作为模板和快速验证入口。
2. 如果仓库里已有旧的 HBGNet 数据集引用，可以增加一个薄兼容层或把旧文件改成指向新的 FTW Dataset 类。
3. 现有 `DInterface` 不需要改动，只有命令行默认值和测试用例需要适配新的数据集名称与参数。

## 验证

实现后至少补两类检查：

1. 单元测试：在临时目录里构造最小的 FTW 目录结构和少量 `.tif` 文件，验证 `DInterface` 能实例化数据集并返回正确形状的 batch。
2. 入口测试：确认 `main.py` 的 parser 能接受新的数据集参数，并且默认参数不影响现有模板行为。

## 风险与处理

1. 如果真实 `.tif` 标签格式和目录约定不一致，Dataset 需要在读取阶段给出更明确的报错信息。
2. 如果后续要支持更多国家或新的 split，当前设计只需要扩展目录扫描逻辑，不需要改训练接口。
3. 如果某些标签不是严格二值或单通道，读取函数需要在不改变主接口的前提下补充归一化规则。

## 结论

本次重构的核心是把 HBGNet 风格的数据读取逻辑收敛成项目内标准 Dataset：一个统一入口、少量清晰参数、按目录自动发现样本、兼容训练和预测。这样既能直接服务 FTW 数据，也能和当前模板保持一致。