import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(img_size, train=True):
    """
    Retorna las transformaciones necesarias. 
    Si train=True, incluye aumento de datos para evitar overfitting.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5), # Aumento de datos: giro aleatorio
            transforms.RandomRotation(15),           # Giro leve
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_data_loaders(data_dir, img_size, batch_size):
    """
    Crea los DataLoaders para train y val asumiendo la estructura de carpetas:
    data/train/clase1... y data/val/clase1...
    """
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_path, transform=get_transforms(img_size, train=True))
    val_dataset = datasets.ImageFolder(val_path, transform=get_transforms(img_size, train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes