# DCGAN Crop Leaf Disease Synthesis - Data Loader
# Plant-specific, Disease-specific dataset handling

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, get_class_folder


class PlantDiseaseDataset(Dataset):
    """
    Dataset for loading plant leaf images filtered by plant type and disease.
    
    Args:
        plant: Plant name (e.g., "Tomato", "Potato")
        disease: Disease type (e.g., "healthy", "Late_blight")
        transform: Optional transform to apply to images
    """
    
    def __init__(self, plant: str, disease: str, transform=None):
        self.plant = plant
        self.disease = disease
        self.transform = transform
        
        # Get the folder name for this plant-disease combination
        folder_name = get_class_folder(plant, disease)
        self.data_path = os.path.join(DATA_DIR, folder_name)
        
        if not os.path.exists(self.data_path):
            raise ValueError(f"Dataset path not found: {self.data_path}")
        
        # Load all image paths
        self.image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
        
        for fname in os.listdir(self.data_path):
            ext = os.path.splitext(fname)[1]
            if ext in valid_extensions:
                self.image_paths.append(os.path.join(self.data_path, fname))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")
        
        print(f"[Dataset] Loaded {len(self.image_paths)} images for {plant} - {disease}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image as RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_transforms():
    """
    Get the standard transforms for DCGAN training.
    - Resize to IMG_SIZE x IMG_SIZE
    - Convert to tensor
    - Normalize to [-1, 1] for tanh output
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range
    ])


def create_dataloader(plant: str, disease: str, batch_size: int = None, 
                      shuffle: bool = True, num_workers: int = None):
    """
    Create a DataLoader for a specific plant-disease combination.
    
    Args:
        plant: Plant name
        disease: Disease type
        batch_size: Batch size (default: from config)
        shuffle: Whether to shuffle
        num_workers: Number of workers (default: from config)
    
    Returns:
        DataLoader
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    transform = get_transforms()
    dataset = PlantDiseaseDataset(plant, disease, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    return dataloader


def create_train_val_dataloaders(plant: str, disease: str, val_split: float = 0.1,
                                  batch_size: int = None, num_workers: int = None):
    """
    Create train and validation DataLoaders with a split.
    
    Args:
        plant: Plant name
        disease: Disease type
        val_split: Fraction for validation (default: 0.1)
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    transform = get_transforms()
    full_dataset = PlantDiseaseDataset(plant, disease, transform=transform)
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"[Split] Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")
    loader = create_dataloader("Tomato", "healthy", batch_size=4)
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    print(f"Value range: [{batch.min():.2f}, {batch.max():.2f}]")
    print("DataLoader test passed!")
