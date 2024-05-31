import os
import matplotlib.pyplot as plt


def count_images_in_classes(data_folder):
    class_counts = {}
    for class_folder in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_folder)
        if os.path.isdir(class_path):
            class_counts[class_folder] = len(
                [
                    file
                    for file in os.listdir(class_path)
                    if file.endswith(".jpg") or file.endswith(".png")
                ]
            )
    return class_counts


def plot_class_distribution(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color="blue")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Images Across Classes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_folder = (
        "processed_data/train"  # Use the folder where your processed images are stored
    )
    class_counts = count_images_in_classes(data_folder)
    plot_class_distribution(class_counts)
