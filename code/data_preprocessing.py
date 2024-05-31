from PIL import Image, ImageEnhance
import os


def resize_image(image_path, output_path, size=(48, 48)):
    """Resizes an image to the specified size and converts it to grayscale."""
    with Image.open(image_path) as img:
        img = img.resize(size)
        img = img.convert("L")  # Convert to grayscale
        img.save(output_path)


def enhance_image(image_path, output_path):
    """Applies light processing to enhance the image."""
    with Image.open(image_path) as img:
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        img.save(output_path)


def process_images(input_folder, output_folder, size=(48, 48)):
    """Processes all images in the input folder and saves them to the output folder."""
    for class_folder in os.listdir(input_folder):
        class_input_folder = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_input_folder) and not class_folder.startswith(
            "."
        ):  # Check if it's a directory and not hidden
            class_output_folder = os.path.join(output_folder, class_folder)
            os.makedirs(class_output_folder, exist_ok=True)

            for filename in os.listdir(class_input_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    input_path = os.path.join(class_input_folder, filename)
                    output_path = os.path.join(class_output_folder, filename)
                    resize_image(input_path, output_path, size)
                    enhance_image(output_path, output_path)


if __name__ == "__main__":
    input_folder = "data/train"
    output_folder = "processed_data/train"

    process_images(input_folder, output_folder)
