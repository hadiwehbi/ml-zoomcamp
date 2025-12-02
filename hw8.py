import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = "./data"
BATCH_SIZE = 20
NUM_EPOCHS = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HairCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        # After conv (no padding): 200 -> 198, then pool: 198 -> 99
        # So feature map: (32, 99, 99)
        self.flatten_dim = 32 * 99 * 99

        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x  # logits


# ---------------------- Data Loaders ----------------------

def create_dataloaders(with_augmentations: bool = False):
    """
    Creates train and test dataloaders.
    When with_augmentations=True, apply extra augmentations to training transforms
    (Question 5 & 6 part).
    """
    base_transforms = [
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]

    if with_augmentations:
        train_transform = transforms.Compose([
            transforms.RandomRotation(50),
            transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            *base_transforms,
        ])
    else:
        train_transform = transforms.Compose(base_transforms)

    test_transform = transforms.Compose(base_transforms)

    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


# ---------------------- Training Utilities ----------------------

def count_parameters(model: nn.Module) -> int:
    """(Question 2)"""
    return sum(p.numel() for p in model.parameters())


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.002, momentum=0.8):
    """
    Generic training loop, as in the homework text.
    Returns history dict with keys 'acc', 'loss', 'val_acc', 'val_loss'.
    """
    criterion = nn.BCEWithLogitsLoss()   # Question 1
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    history = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)  # shape (batch, 1)

            optimizer.zero_grad()
            outputs = model(images)        # logits
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

        # ---- Validation ----
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        val_epoch_loss = val_running_loss / total_val
        val_epoch_acc = correct_val / total_val
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"loss={epoch_loss:.4f} acc={epoch_acc:.4f} "
            f"val_loss={val_epoch_loss:.4f} val_acc={val_epoch_acc:.4f}"
        )

    return history


def evaluate_on_test(model, test_loader):
    """Return (test_loss, test_accuracy) on the test set."""
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc


# ---------------------- Main Script ----------------------

def main():
    set_seed(SEED)

    print("Using device:", DEVICE)

    # ---------- Load data (no augmentations) ----------
    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(with_augmentations=False)

    # ---------- Initialize model ----------
    model = HairCNN().to(DEVICE)
    total_params = count_parameters(model)
    print("Total parameters (Q2):", total_params)

    # ---------- Train for 10 epochs (Questions 3 & 4) ----------
    print("\nTraining baseline model (no augmentations) for 10 epochs...")
    history = train_model(model, train_loader, test_loader, num_epochs=NUM_EPOCHS)

    # Question 3: median train accuracy over epochs
    median_train_acc = float(np.median(history["acc"]))
    print("Q3 - Median training accuracy:", median_train_acc)

    # Question 4: std of training loss over epochs
    std_train_loss = float(np.std(history["loss"]))
    print("Q4 - Std of training loss:", std_train_loss)

    # ---------- Data augmentation and more training (Questions 5 & 6) ----------
    print("\nReloading data with augmentations for training...")
    train_dataset_aug, test_dataset_aug, train_loader_aug, test_loader_aug = create_dataloaders(
        with_augmentations=True
    )

    # Typically, we continue training the same model for 10 more epochs.
    print("Continuing training with augmentations for 10 more epochs...")
    history_aug = train_model(model, train_loader_aug, test_loader_aug, num_epochs=NUM_EPOCHS)

    # Question 5: mean test loss over all 10 augmented epochs (history_aug["val_loss"])
    mean_test_loss_aug = float(np.mean(history_aug["val_loss"]))
    print("Q5 - Mean test loss with augmentations:", mean_test_loss_aug)

    # Question 6: average test accuracy for last 5 epochs (6–10) with augmentations
    last5_acc = history_aug["val_acc"][5:10]
    avg_last5_acc = float(np.mean(last5_acc))
    print("Q6 - Average test accuracy (epochs 6–10, augmented):", avg_last5_acc)


if __name__ == "__main__":
    main()
