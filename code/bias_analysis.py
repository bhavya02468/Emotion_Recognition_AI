import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import MainCNN

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the model
model = MainCNN()
model.load_state_dict(torch.load('best_MainCNN_Model.pth'))
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Load datasets for bias analysis
biased_data_root = 'biased_data_test'

male_dataset = datasets.ImageFolder(root=f'{biased_data_root}/male', transform=transform)
female_dataset = datasets.ImageFolder(root=f'{biased_data_root}/female', transform=transform)

# Create DataLoaders if datasets are not empty
batch_size = 1
male_loader = DataLoader(male_dataset, batch_size=batch_size, shuffle=False) if male_dataset else None
female_loader = DataLoader(female_dataset, batch_size=batch_size, shuffle=False) if female_dataset else None

# Define a function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

# Evaluate the model for each group if data is available
if male_loader:
    male_metrics = evaluate_model(model, male_loader, device)
else:
    male_metrics = (0, 0, 0, 0)

if female_loader:
    female_metrics = evaluate_model(model, female_loader, device)
else:
    female_metrics = (0, 0, 0, 0)

# Print Bias Analysis Table
print("Bias Analysis Table:")
print(f"{'Group':<10}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
print(f"{'Male':<10}{male_metrics[0]:<10.4f}{male_metrics[1]:<10.4f}{male_metrics[2]:<10.4f}{male_metrics[3]:<10.4f}")
print(f"{'Female':<10}{female_metrics[0]:<10.4f}{female_metrics[1]:<10.4f}{female_metrics[2]:<10.4f}{female_metrics[3]:<10.4f}")
