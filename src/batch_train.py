# DCGAN Batch Training Script with Transfer Learning
# Automates grouped training across similar plant species

import os
import sys
import argparse
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, PLANT_DISEASE_MAP,
    STAGE1_LR, STAGE2_LR
)
from src.models import Generator, Discriminator
from src.data_loader import create_dataloader
from src.utils import load_generator_only, find_latest_checkpoint
from src.train import train_stage

# Plant groups based on leaf morphology for transfer learning
PLANT_GROUPS = {
    "solanaceae": {
        "plants": ["Tomato", "Potato", "Pepper"],
        "base_plant": "Tomato",
        "description": "Compound leaves, same family"
    },
    "rosaceae": {
        "plants": ["Apple", "Cherry", "Peach"],
        "base_plant": "Apple",
        "description": "Simple oval leaves with serrated edges"
    },
    "berry": {
        "plants": ["Strawberry", "Raspberry"],
        "base_plant": "Strawberry",
        "description": "Trifoliate compound leaves"
    },
    "vine": {
        "plants": ["Grape", "Squash"],
        "base_plant": "Grape",
        "description": "Large lobed palmate leaves"
    },
    "unique": {
        "plants": ["Corn", "Orange", "Blueberry", "Soybean"],
        "base_plant": None,
        "description": "Unique morphology, train from scratch"
    }
}

# Epoch configuration
EPOCH_CONFIG = {
    "base_healthy": 330,      # Optimal epochs (quality degrades after this)
    "transfer_healthy": 100,  # Epochs for transferred healthy (fine-tune)
    "disease": 150,           # Epochs per disease
}


def get_plant_group(plant: str) -> tuple:
    """Get the group and base plant for a given plant."""
    for group_name, group_info in PLANT_GROUPS.items():
        if plant in group_info["plants"]:
            return group_name, group_info["base_plant"]
    return "unique", None


def train_plant_full(plant: str, device: str = 'cuda', 
                     base_checkpoint: str = None,
                     healthy_epochs: int = None,
                     disease_epochs: int = None):
    """
    Train all diseases for a single plant.
    
    Args:
        plant: Plant name
        device: Device to use
        base_checkpoint: Optional base checkpoint to start from (for transfer learning)
        healthy_epochs: Override healthy epochs
        disease_epochs: Override per-disease epochs
    """
    if plant not in PLANT_DISEASE_MAP:
        print(f"Error: Unknown plant '{plant}'")
        return
    
    diseases = PLANT_DISEASE_MAP[plant]
    group_name, base_plant = get_plant_group(plant)
    
    print(f"\n{'='*60}")
    print(f"Training {plant} (Group: {group_name})")
    print(f"Diseases: {diseases}")
    print(f"{'='*60}\n")
    
    # Determine epochs
    is_transfer = base_checkpoint is not None
    if healthy_epochs is None:
        healthy_epochs = EPOCH_CONFIG["transfer_healthy"] if is_transfer else EPOCH_CONFIG["base_healthy"]
    if disease_epochs is None:
        disease_epochs = EPOCH_CONFIG["disease"]
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Load base checkpoint for transfer learning
    if base_checkpoint:
        print(f"Loading base model from: {base_checkpoint}")
        load_generator_only(base_checkpoint, generator, device)
    
    # Stage 1: Train on healthy
    if "healthy" in diseases:
        print(f"\n[Stage 1] Training {plant} healthy for {healthy_epochs} epochs")
        
        # Check for existing checkpoint to resume
        resume_checkpoint = find_latest_checkpoint(plant, "healthy")
        
        healthy_loader = create_dataloader(plant, "healthy", batch_size=BATCH_SIZE)
        train_stage(
            generator, discriminator, healthy_loader, device,
            plant, "healthy", stage=1,
            epochs=healthy_epochs, lr=STAGE1_LR,
            resume_from=resume_checkpoint
        )
    
    # Stage 2: Train each disease
    for disease in diseases:
        if disease == "healthy":
            continue
        
        print(f"\n[Stage 2] Training {plant} {disease} for {disease_epochs} epochs")
        
        # Reinitialize discriminator for each disease
        discriminator = Discriminator().to(device)
        
        # Check for existing checkpoint
        resume_checkpoint = find_latest_checkpoint(plant, disease)
        
        disease_loader = create_dataloader(plant, disease, batch_size=BATCH_SIZE)
        train_stage(
            generator, discriminator, disease_loader, device,
            plant, disease, stage=2,
            epochs=disease_epochs, lr=STAGE2_LR,
            resume_from=resume_checkpoint
        )
    
    print(f"\n{'='*60}")
    print(f"Completed training for {plant}!")
    print(f"{'='*60}\n")


def train_group(group_name: str, device: str = 'cuda'):
    """
    Train all plants in a group using transfer learning.
    
    Args:
        group_name: Name of the plant group
        device: Device to use
    """
    if group_name not in PLANT_GROUPS:
        print(f"Error: Unknown group '{group_name}'")
        print(f"Available groups: {list(PLANT_GROUPS.keys())}")
        return
    
    group_info = PLANT_GROUPS[group_name]
    plants = group_info["plants"]
    base_plant = group_info["base_plant"]
    
    print(f"\n{'#'*60}")
    print(f"# Training Group: {group_name.upper()}")
    print(f"# Plants: {plants}")
    print(f"# Base Plant: {base_plant}")
    print(f"# {group_info['description']}")
    print(f"{'#'*60}\n")
    
    # Train base plant first (or each plant from scratch for unique group)
    if base_plant is None:
        # Unique group - train each from scratch
        for plant in plants:
            train_plant_full(plant, device)
    else:
        # Train base plant
        train_plant_full(base_plant, device)
        
        # Get base checkpoint
        base_checkpoint = find_latest_checkpoint(base_plant, "healthy")
        
        # Transfer to other plants in group
        for plant in plants:
            if plant == base_plant:
                continue
            
            print(f"\n{'*'*60}")
            print(f"* Transferring from {base_plant} to {plant}")
            print(f"{'*'*60}\n")
            
            train_plant_full(plant, device, base_checkpoint=base_checkpoint)
    
    print(f"\n{'#'*60}")
    print(f"# Group {group_name.upper()} training complete!")
    print(f"{'#'*60}\n")


def train_all(device: str = 'cuda'):
    """Train all plant groups sequentially."""
    print("\n" + "="*60)
    print("FULL DATASET TRAINING WITH TRANSFER LEARNING")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    
    # Train groups in order (largest variety first for better base models)
    group_order = ["solanaceae", "rosaceae", "vine", "berry", "unique"]
    
    for group_name in group_order:
        train_group(group_name, device)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print(f"Total time: {duration}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='DCGAN Batch Training with Transfer Learning')
    
    parser.add_argument('--mode', type=str, default='plant',
                       choices=['plant', 'group', 'all'],
                       help='Training mode: single plant, group, or all')
    parser.add_argument('--plant', type=str, default=None,
                       help='Plant name for single plant training')
    parser.add_argument('--group', type=str, default=None,
                       choices=list(PLANT_GROUPS.keys()),
                       help='Group name for group training')
    parser.add_argument('--base-checkpoint', type=str, default=None,
                       help='Base checkpoint for transfer learning')
    parser.add_argument('--healthy-epochs', type=int, default=None,
                       help='Override healthy training epochs')
    parser.add_argument('--disease-epochs', type=int, default=None,
                       help='Override disease training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--list-groups', action='store_true',
                       help='List all plant groups and exit')
    
    args = parser.parse_args()
    
    # List groups
    if args.list_groups:
        print("\nPlant Groups for Transfer Learning:\n")
        for name, info in PLANT_GROUPS.items():
            print(f"  {name}:")
            print(f"    Plants: {info['plants']}")
            print(f"    Base: {info['base_plant']}")
            print(f"    Description: {info['description']}\n")
        return
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run training
    if args.mode == 'all':
        train_all(args.device)
    
    elif args.mode == 'group':
        if args.group is None:
            print("Error: --group required for group training")
            return
        train_group(args.group, args.device)
    
    elif args.mode == 'plant':
        if args.plant is None:
            print("Error: --plant required for plant training")
            return
        train_plant_full(
            args.plant, args.device,
            base_checkpoint=args.base_checkpoint,
            healthy_epochs=args.healthy_epochs,
            disease_epochs=args.disease_epochs
        )


if __name__ == "__main__":
    main()
