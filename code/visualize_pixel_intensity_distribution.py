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


def plot_pixel_intensity_distribution(class_intensity_distributions):
    plt.figure(figsize=(12, 8))
    for class_name, intensities in class_intensity_distributions.items():
        plt.hist(intensities, bins=256, alpha=0.5, label=class_name, density=True)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Pixel Intensity Distribution per Class")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_folder = "data/train"  # Use the folder where your processed images are stored
    class_intensity_distributions = get_pixel_intensity_distribution(data_folder)
    plot_pixel_intensity_distribution(class_intensity_distributions)
