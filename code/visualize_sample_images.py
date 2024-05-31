import os
import random
import matplotlib.pyplot as plt
from PIL import Image


def show_sample_images_side_by_side(data_folder, samples_per_class=5):
    classes = [
        d
        for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ]
    num_classes = len(classes)
    plt.figure(figsize=(15, num_classes * 2))

    for i, class_name in enumerate(classes):
        class_path = os.path.join(data_folder, class_name)
        image_files = [
            f
            for f in os.listdir(class_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        if len(image_files) > samples_per_class:
            image_files = random.sample(image_files, samples_per_class)
        else:
            image_files = image_files[:samples_per_class]

        for j, image_file in enumerate(image_files):
            image_path = os.path.join(class_path, image_file)
            with Image.open(image_path) as img:
                plt.subplot(
                    num_classes, samples_per_class, i * samples_per_class + j + 1
                )
                plt.imshow(img, cmap="gray")
                plt.axis("off")
                if j == 0:
                    plt.text(
                        -20,
                        img.size[1] // 2,
                        class_name,
                        va="center",
                        ha="right",
                        size="large",
                        rotation=90,
                    )

    plt.tight_layout(
        rect=[0.1, 0, 1, 1]
    )  # Adjust the rectangle parameter to make space for labels
    plt.show()


if __name__ == "__main__":
    data_folder = (
        "processed_data/train"  # Use the folder where your processed images are stored
    )
    show_sample_images_side_by_side(data_folder, samples_per_class=5)
