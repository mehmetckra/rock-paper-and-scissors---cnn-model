import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import gradio as gr
import os

# Model ve parametreler
DATA_DIR = "./archive"
IMG_SIZE = 128
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sınıf isimlerini otomatik al
# class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
class_names = ['kağıt', 'taş', 'makas'] # Alfabetik sıraya göre (paper, rock, scissors)
# Model tanımı (main.py ile aynı olmalı)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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

# Modeli yükle
model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# Görseli modele uygun hale getiren transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

def predict(image):
    if image is None:
        return "Lütfen bir görsel yükleyin."
    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs, 1).item()
    return f"Tahmin edilen sınıf: {class_names[pred]}"

# Gradio arayüzü
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Bir Taş, Kağıt veya Makas görseli yükleyin"),
    outputs=gr.Textbox(label="Tahmin"),
    title="Taş-Kağıt-Makas Sınıflandırma",
    description="Bir taş, kağıt veya makas el hareketi görseli yükleyin ve modelin tahminini görün."
)

if __name__ == "__main__":
    iface.launch()
