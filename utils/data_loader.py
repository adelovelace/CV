import os
import kagglehub
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def download_kaggle_datasets(data_dir='./data'):
    # Let kagglehub manage caching directly; it won't re-download if already present.
    print("Checking FER2013 dataset...")
    fer_path = kagglehub.dataset_download("msambare/fer2013")
    
    print("Checking CK+ dataset...")
    ck_path = kagglehub.dataset_download("shawon10/ckplus")
    
    # The CK+ dataset from this specific Kaggle repo extracts into a subfolder called 'CK+48'
    if os.path.exists(os.path.join(ck_path, 'CK+48')):
        ck_path = os.path.join(ck_path, 'CK+48')
        
    return fer_path, ck_path

def get_transforms(img_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_data_loaders(dataset_path, img_size, batch_size, seed=42):
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'test')

    # Case 1: Dataset is already pre-split into 'train' and 'test' (e.g., FER2013)
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[!] Found pre-split folders in {dataset_path}")
        train_dataset = datasets.ImageFolder(train_path, transform=get_transforms(img_size, train=True))
        val_dataset = datasets.ImageFolder(val_path, transform=get_transforms(img_size, train=False))
        class_names = train_dataset.classes
        
    # Case 2: Dataset is a single folder of classes and needs dynamic splitting (e.g., CK+)
    else:
        print(f"[!] No pre-split folders found. Dynamically splitting {dataset_path}...")
        
        # Load the entire dataset
        full_dataset = datasets.ImageFolder(dataset_path)
        class_names = full_dataset.classes
        
        # Calculate split sizes (80% Train, 20% Validation)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Perform the random split
        train_subset, val_subset = random_split(
            full_dataset, 
            [train_size, val_size],
            # Pass the dynamic seed into the generator 
            generator=torch.Generator().manual_seed(seed) 
        )
        
        # Apply the respective transformations to the subsets
        train_dataset = CustomDatasetWrapper(train_subset, transform=get_transforms(img_size, train=True))
        val_dataset = CustomDatasetWrapper(val_subset, transform=get_transforms(img_size, train=False))

    # Initialize the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names


# Helper class to apply distinct transforms to random_split subsets
class CustomDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)