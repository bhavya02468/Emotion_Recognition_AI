# project/code/evaluate.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from model import MainCNN, Variant1CNN, Variant2CNN

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the trained model
print("Select the model to evaluate:")
print("1: MainCNN (Main Model)")
print("2: Variant1CNN (Additional Conv Layer)")
print("3: Variant2CNN (Different Kernel Sizes)")
choice = input("Enter 1, 2, or 3: ")

if choice == '1':
    model = MainCNN()
    model_name = "MainCNN"
elif choice == '2':
    model = Variant1CNN()
    model_name = "Variant1CNN"
elif choice == '3':
    model = Variant2CNN()
    model_name = "Variant2CNN"
else:
    raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

model.load_state_dict(torch.load(f'best_{model_name}_Model.pth'))
model.to(device)
model.eval()

# Data loader for evaluation
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(root='../data/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model on the complete dataset
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.append(predicted.item())
        all_labels.append(labels.item())

# Generate confusion matrix and classification report
cm = confusion_matrix(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
print('Confusion Matrix')
print(cm)

print('Classification Report')
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, zero_division=0))
print(f'Accuracy: {accuracy:.4f}')

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model on a single image
def evaluate_single_image(image_path):
    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None

    image = test_transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Example usage
image_path = '../processed_data/test/angry/35665.png'  # Ensure this path is correct
predicted_class = evaluate_single_image(image_path)
if predicted_class is not None:
    print(f'Predicted class for {image_path}: {test_dataset.classes[predicted_class]}')
