"""
Data loading for 2-class classification (single GPU)
Supports pre-split train/val structure
File: ViT-pytorch/data_utils_simple.py
"""
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_loader(args):
    """
    Load custom image dataset for 2-class classification
    
    Expected folder structure:
        args.data_root/
        ├── train/
        │   ├── ants/
        │   └── bees/
        └── val/
            ├── ants/
            └── bees/
    
    Args:
        args: Namespace with attributes:
            - data_root: str, path to dataset root
            - img_size: int, image resolution (default 224)
            - train_batch_size: int, training batch size
            - eval_batch_size: int, validation batch size
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    
    # Training transformations with data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Validation transformations (no augmentation)
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Load training dataset
    print(f"\n📂 Loading training data from: {args.data_root}/train")
    trainset = datasets.ImageFolder(
        root=f'{args.data_root}/train',
        transform=transform_train
    )
    
    # Load validation dataset
    print(f"📂 Loading validation data from: {args.data_root}/val")
    valset = datasets.ImageFolder(
        root=f'{args.data_root}/val',
        transform=transform_val
    )
    
    # Print dataset information
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    print(f"Training samples:   {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    print(f"Classes:            {trainset.classes}")
    print(f"Class to index:     {trainset.class_to_idx}")
    
    # Count samples per class in training set
    class_counts = {}
    for _, label in trainset:
        class_name = trainset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    print(f"\nTraining distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Count samples per class in validation set
    val_class_counts = {}
    for _, label in valset:
        class_name = valset.classes[label]
        val_class_counts[class_name] = val_class_counts.get(class_name, 0) + 1
    print(f"\nValidation distribution:")
    for class_name, count in val_class_counts.items():
        print(f"  {class_name}: {count} images")
    
    print("="*60 + "\n")
    
    # Create DataLoaders for single GPU
    train_loader = DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        valset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Created DataLoaders:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}\n")
    
    return train_loader, val_loader