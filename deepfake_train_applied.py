import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# Config (adjust DATA_DIR to your dataset path)
IMAGE_SIZE = 224  # use 224 for ImageNet models
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_resnet18.pth"
DATA_DIR = os.environ.get("DEEPU_DATA_DIR", "C:/Users/Rituraj Dusane/Desktop/deepfake image detection project/dataset")

# Pretrained weights metadata for normalization
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
MEAN, STD = WEIGHTS.meta["mean"], WEIGHTS.meta["std"]

# Data Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Load Datasets
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

num_classes = len(train_data.classes)

# Class balancing with WeightedRandomSampler
class_counts = np.bincount(train_data.targets)
class_counts = class_counts if len(class_counts) > 0 else np.array([1] * max(2, num_classes))
class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
sample_weights = class_weights[train_data.targets]
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

pin_memory = torch.cuda.is_available()

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=pin_memory)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin_memory)

# Model: ResNet18 transfer learning
model = resnet18(weights=WEIGHTS)
# Replace classifier head
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)
model = model.to(DEVICE)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# OneCycleLR for stable convergence
steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.15,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# Warmup: freeze backbone for first few epochs, then unfreeze
def set_backbone_requires_grad(m, requires_grad: bool):
    for name, p in m.named_parameters():
        if not name.startswith("fc"):
            p.requires_grad = requires_grad

WARMUP_EPOCHS = 3
set_backbone_requires_grad(model, False)

# Training Loop with early stopping
best_accuracy = 0.0
patience = 5
no_improve_epochs = 0

for epoch in range(EPOCHS):
    model.train()

    # Unfreeze after warmup
    if epoch == WARMUP_EPOCHS:
        set_backbone_requires_grad(model, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            epochs=EPOCHS - epoch,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )

    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    # Validation
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    avg_train_loss = running_loss / len(train_loader.dataset)
    avg_val_loss = val_loss / len(val_loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f} | F1: {f1:.4f}"
    )

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), MODEL_PATH)
        best_accuracy = accuracy
        no_improve_epochs = 0
        print("✅ Saved Best Model")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("⏹️ Early stopping")
            break

print(f"\nTraining complete. Best Validation Accuracy: {best_accuracy:.4f}")

# Load best checkpoint and print a classification report
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
all_true, all_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())

print(classification_report(all_true, all_pred, target_names=val_data.classes, digits=4))