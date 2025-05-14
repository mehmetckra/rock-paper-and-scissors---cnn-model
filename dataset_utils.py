import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_transforms(img_size: int = 128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return train_transform, val_transform

def get_class_names(data_dir: str):
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 128,
    val_split: float = 0.2,
    seed: int = 42
):
    train_transform, val_transform = get_transforms(img_size)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = full_dataset.classes

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=None if seed is None else torch.Generator().manual_seed(seed)
    )

    # Validation set için transform'u değiştir
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names
