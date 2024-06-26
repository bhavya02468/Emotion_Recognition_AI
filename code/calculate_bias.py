# project/code/calculate_bias.py
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from model import MainCNN, Variant1CNN, Variant2CNN

# Custom dataset class to handle the specific directory structure
class CustomImageDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name, group = self.samples[idx]
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[class_name], group

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the trained model
print("Select the model to evaluate:")
print("1: MainCNN (Main Model)")
print("2: Variant1CNN (Additional Conv Layer)")
print("3: Variant2CNN (Different Kernel Sizes)")
choice = input("Enter 1, 2, or 3: ")

all_preds = []
all_labels = []
all_groups = []

if choice == '1':
    model = MainCNN()
    model_name = "MainCNN"
    model.load_state_dict(torch.load(f'best_{model_name}_Model.pth'))
    model.to(device)
    model.eval()
elif choice == '2':
    model = Variant1CNN()
    model_name = "Variant1CNN"
    model.load_state_dict(torch.load(f'best_{model_name}_Model.pth'))
    model.to(device)
    model.eval()
elif choice == '3':
    model = Variant2CNN()
    model_name = "Variant2CNN"
    model.load_state_dict(torch.load(f'best_{model_name}_Model.pth'))
    model.to(device)
    model.eval()
else:
    raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

# Data loader for evaluation
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Convert relative paths to absolute paths
base_path = os.path.abspath('../biased_data/test')

# Create a mapping from class names to indices
class_names = ['angry', 'focused', 'happy', 'neutral']
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Load all images and store their paths, class, and group
samples = []
for class_name in class_names:
    for gender in ['male', 'female']:
        group_path = os.path.join(base_path, class_name, gender)
        if os.path.exists(group_path):
            for img_name in os.listdir(group_path):
                if img_name.endswith(('.jpg', '.png')):
                    samples.append((os.path.join(group_path, img_name), class_name, f"{class_name}/{gender}"))

# Load the entire test dataset using the custom dataset class
test_dataset = CustomImageDataset(samples=samples, class_to_idx=class_to_idx, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model on the complete dataset
with torch.no_grad():
    for images, labels, groups in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.append(predicted.item())
        all_labels.append(labels.item())
        all_groups.append(groups[0])

# Ask user for which class bias to calculate
print("Select the class for which you want to see the bias:")
for idx, class_name in enumerate(class_names):
    print(f"{idx+1}: {class_name}")
class_choice = int(input("Enter the number corresponding to the class: "))

if class_choice < 1 or class_choice > len(class_names):
    raise ValueError("Invalid choice. Please enter a valid number.")

selected_class = class_names[class_choice - 1]

# Calculate metrics for each group
group_metrics = {group: {'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0} for group in ['angry/male', 'angry/female', 'focused/male', 'focused/female', 'happy/male', 'happy/female', 'neutral/male', 'neutral/female']}
for group in group_metrics.keys():
    group_indices = [i for i, group_name in enumerate(all_groups) if group_name == group]
    group_preds = [all_preds[i] for i in group_indices]
    group_labels = [all_labels[i] for i in group_indices]
    if group_preds:
        report = classification_report(group_labels, group_preds, labels=[class_to_idx[selected_class]], target_names=[selected_class], output_dict=True, zero_division=0)
        if selected_class in report:
            for metric in ['precision', 'recall', 'f1-score']:
                group_metrics[group][metric] = report[selected_class][metric]
            group_metrics[group]['accuracy'] = accuracy_score(group_labels, group_preds)
    else:
        print(f'No images found for group: {group}')

# Calculate and display the bias for the selected class
def calculate_bias(metrics_male, metrics_female, metric_name):
    print(f"Bias in {metric_name} for class {selected_class}:")
    male_value = metrics_male[metric_name]
    female_value = metrics_female[metric_name]
    bias = male_value - female_value
    print(f"{selected_class}: Male = {male_value:.4f}, Female = {female_value:.4f}, Bias = {bias:.4f}")

metrics_male = {'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}
metrics_female = {'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}

num_male_groups = 0
num_female_groups = 0

for group, metrics in group_metrics.items():
    if 'male' in group:
        for metric in ['precision', 'recall', 'f1-score', 'accuracy']:
            metrics_male[metric] += metrics[metric]
        num_male_groups += 1
    else:
        for metric in ['precision', 'recall', 'f1-score', 'accuracy']:
            metrics_female[metric] += metrics[metric]
        num_female_groups += 1

# Average the metrics
for metric in ['precision', 'recall', 'f1-score', 'accuracy']:
    if num_male_groups > 0:
        metrics_male[metric] /= num_male_groups
    if num_female_groups > 0:
        metrics_female[metric] /= num_female_groups

# Display the selected class bias
for metric in ['precision', 'recall', 'f1-score', 'accuracy']:
    calculate_bias(metrics_male, metrics_female, metric)
