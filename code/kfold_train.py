import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import MainCNN, Variant1CNN, Variant2CNN, FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Hyperparameters
num_epochs = 100
learning_rate = 0.01
batch_size = 64
momentum = 0.9
weight_decay = 1e-4
patience = 10

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Load dataset
full_dataset = datasets.ImageFolder(root='../data/train', transform=transform)

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Placeholder for performance metrics
fold_metrics = {
    'accuracy': [],
    'macro_precision': [],
    'macro_recall': [],
    'macro_f1': [],
    'micro_precision': [],
    'micro_recall': [],
    'micro_f1': []
}

# Initialize Focal Loss
criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean').to(device)

# Prompt for model selection
print("Select the model to train:")
print("1: MainCNN (Main Model)")
print("2: Variant1CNN (Additional Conv Layer)")
print("3: Variant2CNN (Different Kernel Sizes)")
choice = input("Enter 1, 2, or 3: ")

if choice == '1':
    model_class = MainCNN
    model_name = "MainCNN"
elif choice == '2':
    model_class = Variant1CNN
    model_name = "Variant1CNN"
elif choice == '3':
    model_class = Variant2CNN
    model_name = "Variant2CNN"
else:
    raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

print(f'Training {model_name}...')

# Function to train and evaluate for each fold
def train_and_evaluate(train_idx, val_idx, fold):
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # Splitting train_subset further into actual training set and validation set
    train_size = int(len(train_subset) * 0.85)
    val_size = len(train_subset) - train_size
    train_subset, val_subset_extra = random_split(train_subset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    val_loader_extra = DataLoader(val_subset_extra, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model = model_class().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    best_val_loss = float('inf')
    early_stopping_counter = 0

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

        print(f'Fold [{fold}], Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_fold_{fold}_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

    # Evaluate on validation set
    model.load_state_dict(torch.load(f'best_fold_{fold}_model.pth'))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    fold_metrics['accuracy'].append(accuracy)
    fold_metrics['macro_precision'].append(macro_precision)
    fold_metrics['macro_recall'].append(macro_recall)
    fold_metrics['macro_f1'].append(macro_f1)
    fold_metrics['micro_precision'].append(micro_precision)
    fold_metrics['micro_recall'].append(micro_recall)
    fold_metrics['micro_f1'].append(micro_f1)
    print(f'Fold {fold} - Accuracy: {accuracy:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}, Micro F1: {micro_f1:.4f}')

# Perform k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f'Starting fold {fold + 1}')
    train_and_evaluate(train_idx, val_idx, fold + 1)

# Average metrics across all folds
avg_accuracy = np.mean(fold_metrics['accuracy'])
avg_macro_precision = np.mean(fold_metrics['macro_precision'])
avg_macro_recall = np.mean(fold_metrics['macro_recall'])
avg_macro_f1 = np.mean(fold_metrics['macro_f1'])
avg_micro_precision = np.mean(fold_metrics['micro_precision'])
avg_micro_recall = np.mean(fold_metrics['micro_recall'])
avg_micro_f1 = np.mean(fold_metrics['micro_f1'])

print(f'Average Accuracy: {avg_accuracy:.4f}')
print(f'Average Macro Precision: {avg_macro_precision:.4f}')
print(f'Average Macro Recall: {avg_macro_recall:.4f}')
print(f'Average Macro F1 Score: {avg_macro_f1:.4f}')
print(f'Average Micro Precision: {avg_micro_precision:.4f}')
print(f'Average Micro Recall: {avg_micro_recall:.4f}')
print(f'Average Micro F1 Score: {avg_micro_f1:.4f}')

# Final testing phase
test_dataset = datasets.ImageFolder(root='../data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_model = model_class().to(device)
best_model.load_state_dict(torch.load(f'best_{model_name}_Model.pth'))
best_model.eval()

all_test_preds = []
all_test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)
        all_test_preds.extend(predicted.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
test_macro_precision = precision_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
test_macro_recall = recall_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
test_macro_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
test_micro_precision = precision_score(all_test_labels, all_test_preds, average='micro', zero_division=0)
test_micro_recall = recall_score(all_test_labels, all_test_preds, average='micro', zero_division=0)
test_micro_f1 = f1_score(all_test_labels, all_test_preds, average='micro', zero_division=0)

print('Final Test Results:')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Macro Precision: {test_macro_precision:.4f}')
print(f'Test Macro Recall: {test_macro_recall:.4f}')
print(f'Test Macro F1 Score: {test_macro_f1:.4f}')
print(f'Test Micro Precision: {test_micro_precision:.4f}')
print(f'Test Micro Recall: {test_micro_recall:.4f}')
print(f'Test Micro F1 Score: {test_micro_f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(all_test_labels, all_test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
