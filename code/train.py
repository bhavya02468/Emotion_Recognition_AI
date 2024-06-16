# project/code/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import MainCNN, Variant1CNN, Variant2CNN, FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Hyperparameters
num_epochs = 100
learning_rate = 0.01  # Reduced learning rate
batch_size = 64
momentum = 0.9
weight_decay = 1e-4
val_split = 0.2  # Validation split ratio (20%)
patience = 10  # Early stopping patience

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data loaders with extensive augmentation
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Load the dataset and split it
full_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply validation transform to the validation dataset
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Focal Loss
criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean').to(device)

# Prompt for model selection
print("Select the model to train:")
print("1: MainCNN (Main Model)")
print("2: Variant1CNN (Additional Conv Layer)")
print("3: Variant2CNN (Different Kernel Sizes)")
choice = input("Enter 1, 2, or 3: ")

if choice == '1':
    model = MainCNN().to(device)
    model_name = "MainCNN"
elif choice == '2':
    model = Variant1CNN().to(device)
    model_name = "Variant1CNN"
elif choice == '3':
    model = Variant2CNN().to(device)
    model_name = "Variant2CNN"
else:
    raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

print(f'Training {model_name}...')

# Initialize optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

best_val_loss = float('inf')
early_stopping_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    scheduler.step(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'best_{model_name}_Model.pth')
        print('Best model saved!')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Early stopping
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break

# Save the final model
torch.save(model.state_dict(), f'final_{model_name}_Model.pth')
print('Final model saved!')
