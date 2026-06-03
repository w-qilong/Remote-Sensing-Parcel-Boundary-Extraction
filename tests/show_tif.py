"""显示单个 TIFF 文件的最简脚本。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from skimage.color import label2rgb

window_a_file = r"ftw_data\ftw_dataset\rwanda\train\image\1592589.tif"
window_b_file = r"ftw_data\ftw_origin_data\ftw\kenya\s2_images\window_b\g0_0000000000-0000008192.tif"
semantic_2_class_file = r"ftw_data\ftw_dataset\rwanda\train\mask\1592589.tif"
semantic_3_class_file = r"ftw_data\ftw_dataset\rwanda\train\boundary\1592589.tif"
instance_class_file = r"ftw_data\ftw_origin_data\ftw\kenya\label_masks\instance\g0_0000000000-0000008192.tif"


def plot_data(window_a_file, window_b_file, semantic_2_class_file, semantic_3_class_file, instance_class_file):
    # Load window A and window B
    with rasterio.open(window_a_file) as src:
        window_a = src.read()[0:3, :, :]  # Reading first 3 bands
        window_a = window_a.transpose(1, 2, 0) / 3000  # Normalizing
    
    with rasterio.open(window_b_file) as src:
        window_b = src.read()[0:3, :, :]  # Reading first 3 bands
        window_b = window_b.transpose(1, 2, 0) / 3000  # Normalizing

    # Load semantic and instance data
    with rasterio.open(semantic_2_class_file) as src:
        semantic_2_class = src.read()

    with rasterio.open(semantic_3_class_file) as src:
        semantic_3_class = src.read()

    with rasterio.open(instance_class_file) as src:
        instance_class = src.read()[0]  # Assuming it's single band data for instance labels

        # Generate random colors for each instance class
        unique_labels = np.unique(instance_class)
        colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in unique_labels]
        instance_mask_rgb = label2rgb(instance_class, bg_label=0, bg_color=(0, 0, 0), colors=colors)


    # Create subplots to visualize the data
    fig, axs = plt.subplots(1, 5, figsize=(20, 10))
    
    # Display Window A
    axs[0].imshow(np.clip(window_a, 0, 1))  # Clipping to avoid over-scaling issues
    axs[0].set_title('Window A')
    
    # Display Window B
    axs[1].imshow(np.clip(window_b, 0, 1))  # Clipping to avoid over-scaling issues
    axs[1].set_title('Window B')
    
    # Display Semantic 2-class
    axs[2].imshow(semantic_2_class[0], cmap='viridis', vmin=0, vmax=2)
    axs[2].set_title('Semantic 2-class')
    
    # Display Semantic 3-class
    axs[3].imshow(semantic_3_class[0], cmap='viridis', vmin=0, vmax=2)
    axs[3].set_title('Semantic 3-class')
    
    # Display Instance class with RGB mask
    axs[4].imshow(instance_mask_rgb)
    axs[4].set_title('Instance class')

    for ax in axs:
        ax.axis('off')

    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_data(window_a_file, window_b_file, semantic_2_class_file, semantic_3_class_file, instance_class_file)