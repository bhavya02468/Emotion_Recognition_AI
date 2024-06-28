import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from torch import autocast
from diffusers import StableDiffusionPipeline
from model import MainCNN

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the model for downstream tasks (MainCNN)
model = MainCNN()
model.load_state_dict(torch.load('best_MainCNN_Model.pth'))
model.to(device)
model.eval()

# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Define root directories
biased_data_root = '../biased_data/train'
augmented_data_root = '../augmented_data'

# Verify the root directory exists
if not os.path.exists(biased_data_root):
    raise FileNotFoundError(f"Source directory '{biased_data_root}' does not exist.")

# Create folders for augmented data if they don't exist
os.makedirs(augmented_data_root, exist_ok=True)
for emotion in ['angry', 'focused', 'happy', 'neutral']:
    os.makedirs(os.path.join(augmented_data_root, emotion), exist_ok=True)

# Function to generate synthetic data using Stable Diffusion
def generate_synthetic_data_diffusion(dest_dir, gender):
    num_images_per_class = 50
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-3", torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    
    synthetic_data = []  # Placeholder for synthetic data
    for emotion in ['angry', 'focused', 'happy', 'neutral']:
        for i in range(num_images_per_class):
            # Generate Stable Diffusion sample for each emotion class
            prompt = f"The face of a {gender} with a {emotion} expression"
            with autocast("cuda"):
                result = pipeline(prompt, guidance_scale=7.5)
                if hasattr(result, "images"):
                    image = result.images[0]  # Assuming only one image is generated
                    image = image.convert("RGB")  # Convert image to RGB mode if necessary
                else:
                    raise KeyError(f"'images' attribute not found in result: {result}")
            
            # Transform image to match dataset expectations (grayscale, resize, tensor)
            image = transform(image)
            
            # Save image to destination directory
            image_path = os.path.join(dest_dir, emotion, f'synthetic_{emotion}_{i}.jpg')
            torchvision.utils.save_image(image, image_path)
            
            synthetic_data.append(image)  # Append to synthetic data list
    
    return synthetic_data

# Function to copy data from source to destination and mix old and new images
def copy_and_mix_data(source_dir, dest_dir):
    for emotion in ['angry', 'focused', 'happy', 'neutral']:
        source_path = os.path.join(source_dir, emotion)
        dest_path = os.path.join(dest_dir, emotion)
        os.makedirs(dest_path, exist_ok=True)
        if os.path.exists(source_path):
            for gender in ['male', 'female']:
                source_gender_path = os.path.join(source_path, gender)
                if os.path.exists(source_gender_path):
                    for filename in os.listdir(source_gender_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            src_file = os.path.join(source_gender_path, filename)
                            dst_file = os.path.join(dest_path, filename)
                            shutil.copyfile(src_file, dst_file)

# User input to select gender for augmentation
selected_gender = input("Enter the gender to augment (male/female): ").strip().lower()

# Determine source directories based on user input
source_dir = biased_data_root
other_gender = 'male' if selected_gender == 'female' else 'female'

# Verify source directories exist
for emotion in ['angry', 'focused', 'happy', 'neutral']:
    source_gender_dir = os.path.join(source_dir, emotion, selected_gender)
    if not os.path.exists(source_gender_dir):
        raise FileNotFoundError(f"Source directory '{source_gender_dir}' does not exist.")
    other_gender_dir = os.path.join(source_dir, emotion, other_gender)
    if not os.path.exists(other_gender_dir):
        raise FileNotFoundError(f"Source directory '{other_gender_dir}' does not exist.")

# Copy data for both genders to augmented data root
copy_and_mix_data(source_dir, augmented_data_root)

# Generate synthetic data using Stable Diffusion
synthetic_data = generate_synthetic_data_diffusion(augmented_data_root, selected_gender)

print(f"Synthetic data saved to {augmented_data_root}.")
print("Data augmentation and synthetic data generation completed.")
