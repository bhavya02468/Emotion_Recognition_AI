import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_pixel_intensity_distribution(data_folder):
    class_intensity_distributions = {}
    for class_folder in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_folder)
        if os.path.isdir(class_path):
            pixel_intensities = []
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(class_path, filename)
                    with Image.open(image_path) as img:
                        img = img.convert("L")  # Ensure the image is in grayscale
                        pixel_intensities.extend(np.array(img).flatten())
            class_intensity_distributions[class_folder] = pixel_intensities
    return class_intensity_distributions


def plot_pixel_intensity_distribution_side_by_side(class_intensity_distributions):
    num_classes = len(class_intensity_distributions)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5), sharey=True)

    if num_classes == 1:
        axes = [axes]  # Make axes iterable if only one class

    for ax, (class_name, intensities) in zip(
        axes, class_intensity_distributions.items()
    ):
        ax.hist(intensities, bins=256, alpha=0.75, density=True)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Pixel Intensity Distribution - {class_name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_folder = (
        "processed_data/train"  # Use the folder where your processed images are stored
    )
    class_intensity_distributions = get_pixel_intensity_distribution(data_folder)
    plot_pixel_intensity_distribution_side_by_side(class_intensity_distributions)
