"""手动查看 FTW TIFF 样本的可视化脚本。

该文件不是自动化测试用例，而是调试数据转换结果时的人工检查工具。路径常量指向
项目内保留的 ``ftw_data`` 数据，运行脚本后会显示影像、二值掩膜、边界和实例标签。
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.color import label2rgb

window_a_file = r"ftw_data/ftw_dataset/rwanda/train/image/1592589.tif"
window_b_file = r"ftw_data/ftw_origin_data/ftw/kenya/s2_images/window_b/g0_0000000000-0000008192.tif"
semantic_2_class_file = r"ftw_data/ftw_dataset/rwanda/train/mask/1592589.tif"
semantic_3_class_file = r"ftw_data/ftw_dataset/rwanda/train/boundary/1592589.tif"
instance_class_file = r"ftw_data/ftw_origin_data/ftw/kenya/label_masks/instance/g0_0000000000-0000008192.tif"


def plot_data(window_a_file, window_b_file, semantic_2_class_file, semantic_3_class_file, instance_class_file):
    """读取五类 FTW 数据并排显示，便于人工确认预处理是否合理。"""

    # 读取两个 Sentinel-2 时间窗口，并只取前三个波段作为 RGB 显示。
    with rasterio.open(window_a_file) as src:
        window_a = src.read()[0:3, :, :]
        window_a = window_a.transpose(1, 2, 0) / 3000
    
    with rasterio.open(window_b_file) as src:
        window_b = src.read()[0:3, :, :]
        window_b = window_b.transpose(1, 2, 0) / 3000

    # 读取语义标签和实例标签，后面会把实例 ID 转成随机颜色方便观察。
    with rasterio.open(semantic_2_class_file) as src:
        semantic_2_class = src.read()

    with rasterio.open(semantic_3_class_file) as src:
        semantic_3_class = src.read()

    with rasterio.open(instance_class_file) as src:
        instance_class = src.read()[0]

        # 每个实例 ID 分配一个随机颜色，背景保持黑色。
        unique_labels = np.unique(instance_class)
        colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in unique_labels]
        instance_mask_rgb = label2rgb(instance_class, bg_label=0, bg_color=(0, 0, 0), colors=colors)


    # 五个子图并排显示，适合快速对比影像与标签是否对齐。
    fig, axs = plt.subplots(1, 5, figsize=(20, 10))
    
    axs[0].imshow(np.clip(window_a, 0, 1))
    axs[0].set_title('Window A')
    
    axs[1].imshow(np.clip(window_b, 0, 1))
    axs[1].set_title('Window B')
    
    axs[2].imshow(semantic_2_class[0], cmap='viridis', vmin=0, vmax=2)
    axs[2].set_title('Semantic 2-class')
    
    axs[3].imshow(semantic_3_class[0], cmap='viridis', vmin=0, vmax=2)
    axs[3].set_title('Semantic 3-class')
    
    axs[4].imshow(instance_mask_rgb)
    axs[4].set_title('Instance class')

    for ax in axs:
        ax.axis('off')

    plt.show()

if __name__ == "__main__":
    plot_data(window_a_file, window_b_file, semantic_2_class_file, semantic_3_class_file, instance_class_file)
