# Run using Python 3.11
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models
import os

# NOTE: not setting seeds so results will vary slightly on each run

# Switch to save or show plots
dpl = False  # set to True to show plots

# Switch to fix all layers but last or train all layers
fix_layers = True  # set to False to train all layers

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

## Either download from dropbox or use local copy
if os.path.exists(os.path.join(script_dir, "xray_data.pt")):
    data_path = os.path.join(script_dir, "xray_data.pt")
    data = torch.load(data_path, weights_only=False)
else:
    dropbox_url = "https://www.dropbox.com/scl/fi/f4dm7jx545iifys32rw3n/xray_data.pt?rlkey=khmvzz8rbkxav66jv5u48bg80&st=goyu8f8w&dl=1"
    response = requests.get(dropbox_url)
    response.raise_for_status() 
    data = torch.load(BytesIO(response.content), weights_only=False)

images = data["images"]
labels = data["labels"]
dataset = TensorDataset(images, labels)

# View some images
# Choose an index
idx = 0
image = images[idx]
label = labels[idx]

# Unnormalize 
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
image = image * std + mean  # reverse normalization
image = image.clamp(0, 1)  # clip values to [0, 1]

# Plot
plt.imshow(image.permute(1, 2, 0))  # CHW -> HWC
plt.title(f"Label: {'Infiltration' if label == 1 else 'No Finding'}")
plt.axis("off")
if dpl:
    plt.show()
else: 
    plt.savefig("xray_sample_image1.png")

# Choose an index
idx = 7
image = images[idx]
label = labels[idx]

# Unnormalize 
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
image = image * std + mean  # reverse normalization
image = image.clamp(0, 1)  # clip values to [0, 1]

# Plot
plt.imshow(image.permute(1, 2, 0))  # CHW -> HWC
plt.title(f"Label: {'Infiltration' if label == 1 else 'No Finding'}")
plt.axis("off")
if dpl:
    plt.show()
else: 
    plt.savefig("xray_sample_image7.png")

# Split data into training and validation sets
train_size = int(0.7 * len(dataset))  
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load pre-trained ResNet18 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # Replace final layer for 2 classes
model = model.to(device)

if fix_layers:
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=3e-5)    

else:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# Display model architecture
print(model)
# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# Trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Training loop
train_losses = []
train_accuracies = []
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")


# Plot training loss and accuracy over epochs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, marker="o")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
if dpl:
    plt.show()
else: 
    plt.savefig("xray_training_metrics.png")

# Evaluation on validation set
model.eval()
val_correct = 0
val_total = 0
all_preds = []
all_labels = []

# Accuracy on validation set
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

val_accuracy = val_correct / val_total
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_correct}/{val_total})")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Finding", "Infiltration"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Validation Set")
if dpl:
    plt.show()
else: 
    plt.savefig("xray_confusion_matrix.png")

