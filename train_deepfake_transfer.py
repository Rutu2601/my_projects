import os
import random
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def build_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    mean, std = weights.meta["mean"], weights.meta["std"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    class_counts = np.bincount(train_dataset.targets)
    class_counts = class_counts if len(class_counts) > 0 else np.array([1, 1])
    class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, num_classes


def build_model(num_classes: int, dropout_p: float = 0.3) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )
    return model


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_targets: list = []
    all_preds: list = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    return avg_loss, acc, f1


def train(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    early_stop_patience: int = 5,
    warmup_epochs: int = 3,
    model_path: str = "best_model_resnet18.pth",
) -> None:
    set_global_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = build_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(num_classes=num_classes, dropout_p=0.3).to(device)

    # Phase 1: freeze backbone for warmup
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=max(0.1, warmup_epochs / max(epochs, 1)),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        # Unfreeze after warmup
        if epoch == warmup_epochs:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=epochs - epoch,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=100.0,
            )

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        train_loss = running_loss / max(total_samples, 1)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }, model_path)
            best_val_acc = val_acc
            epochs_no_improve = 0
            print("Saved new best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # Final evaluation and report
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

    print(f"\nBest checkpoint -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Detailed classification report
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy().tolist())

    print(classification_report(all_targets, all_preds, digits=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake Detection Training - Transfer Learning (ResNet18)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing 'train' and 'val' subfolders")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--model_path", type=str, default="best_model_resnet18.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        early_stop_patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        model_path=args.model_path,
    )