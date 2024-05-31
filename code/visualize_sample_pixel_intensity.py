import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_sample_pixel_intensity(data_folder, samples_per_class=5):
    sample_intensities = {}
    for class_folder in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_folder)
        if os.path.isdir(class_path):
            pixel_intensities = []
            image_files = [
                f
                for f in os.listdir(class_path)
                if f.endswith(".jpg") or f.endswith(".png")
            ][:samples_per_class]
            for filename in image_files:
                image_path = os.path.join(class_path, filename)
                with Image.open(image_path) as img:
                    img = img.convert("L")  # Ensure the image is in grayscale
                    pixel_intensities.append(np.array(img).flatten())
            sample_intensities[class_folder] = pixel_intensities
    return sample_intensities


def plot_sample_pixel_intensity(sample_intensities):
    num_classes = len(sample_intensities)
    samples_per_class = len(next(iter(sample_intensities.values())))
    plt.figure(figsize=(15, num_classes * 2))

    for i, (class_name, intensity_list) in enumerate(sample_intensities.items()):
        for j, intensities in enumerate(intensity_list):
            plt.subplot(num_classes, samples_per_class, i * samples_per_class + j + 1)
            plt.hist(intensities, bins=256, alpha=0.75, density=True)
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.title(f"{class_name} Sample {j + 1}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_folder = (
        "processed_data/train"  # Use the folder where your processed images are stored
    )
    sample_intensities = get_sample_pixel_intensity(data_folder)
    plot_sample_pixel_intensity(sample_intensities)
