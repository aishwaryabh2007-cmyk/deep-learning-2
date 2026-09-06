import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Dataset Path
DATASET_PATH = "dataset/PlantVillage"

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Dataset
dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

# Save class names
with open("class_names.txt", "w") as f:
    for class_name in dataset.classes:
        f.write(class_name + "\n")

# Data Loader
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

# Load Pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(dataset.classes)
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 1

for epoch in range(EPOCHS):

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

      if batch_idx % 20 == 0:
        print(f"Processing Batch {batch_idx}")

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {running_loss:.4f}"
    )

# Save Model
torch.save(
    model.state_dict(),
    "model/disease_model.pth"
)

print("Model Saved Successfully!")