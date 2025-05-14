import torch
import torch.nn as nn
import torch.optim as optim
from dataset_utils import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import random

DATA_DIR = "./archive"
BATCH_SIZE = 32
IMG_SIZE = 128
NUM_CLASSES = 3
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basit CNN modeli
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 128x128x3 -> 128x128x32
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 128x128x32 -> 64x64x32

            nn.Conv2d(32, 64, 3, padding=1), # 64x64x32 -> 64x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 64x64x64 -> 32x32x64

            nn.Conv2d(64, 128, 3, padding=1),# 32x32x64 -> 32x32x128
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 32x32x128 -> 16x16x128
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*16*128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train():
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Eğitim
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Doğrulama
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print("En iyi doğrulama doğruluğu:", best_val_acc)
    return train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == "__main__":
    print(f"Kullanılan cihaz: {DEVICE}") # Bu satırı ekleyin
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        val_split=0.2,
        seed=42
    )

    print("Sınıf isimleri:", class_names)
    print("Eğitim batch sayısı:", len(train_loader))
    print("Doğrulama batch sayısı:", len(val_loader))

    model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, train_accuracies, val_accuracies = train()

    # 1. Eğitim ve doğrulama loss grafiği
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Eğitim ve Doğrulama Loss Grafiği")
    plt.legend()
    plt.show()

    # 2. Eğitim ve doğrulama accuracy grafiği
    plt.figure(figsize=(8,5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Eğitim ve Doğrulama Accuracy Grafiği")
    plt.legend()
    plt.show()

    # 3. Kaydedilen modeli yükle
    best_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
    best_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    best_model.eval()

    # 4. Validation setinden rastgele örnekler al ve tahmin et
    def imshow(img, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
        img = img.cpu().numpy().transpose((1,2,0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    # Validation dataset'e erişim
    val_dataset = val_loader.dataset
    indices = random.sample(range(len(val_dataset)), 8)
    plt.figure(figsize=(16,8))
    for i, idx in enumerate(indices):
        img, label = val_dataset[idx]
        input_img = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = best_model(input_img)
            pred = torch.argmax(output, 1).item()
        plt.subplot(2,4,i+1)
        plt.imshow(imshow(img))
        plt.axis('off')
        plt.title(f"Tahmin: {class_names[pred]}\nGerçek: {class_names[label]}", fontsize=10)
    plt.tight_layout()
    plt.show()
