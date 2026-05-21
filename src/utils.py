# DCGAN Crop Leaf Disease Synthesis - Utility Functions
# Checkpointing, image saving, and visualization

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.utils import make_grid

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, SAMPLES_DIR, LOGS_DIR


def denormalize(tensor):
    """
    Convert tensor from [-1, 1] to [0, 255] for display/saving.
    
    Args:
        tensor: Image tensor in [-1, 1] range
    
    Returns:
        Numpy array in [0, 255] range
    """
    # Move to CPU if needed
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Denormalize: [-1, 1] → [0, 1] → [0, 255]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
    elif tensor.dim() == 3:  # Single image
        tensor = tensor.permute(1, 2, 0).numpy()  # (H, W, C)
    
    return (tensor * 255).astype(np.uint8)


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                    epoch: int, plant: str, disease: str, stage: int,
                    d_loss: float = 0, g_loss: float = 0, metadata: dict = None):
    """
    Save model checkpoint.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        epoch: Current epoch
        plant: Plant name
        disease: Disease type
        stage: Training stage (1 or 2)
        d_loss: Discriminator loss
        g_loss: Generator loss
        metadata: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'plant': plant,
        'disease': disease,
        'stage': stage,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'd_loss': d_loss,
        'g_loss': g_loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Create filename
    stage_name = "healthy" if stage == 1 else disease
    filename = f"G_{plant}_{stage_name}_epoch{epoch:03d}.pth"
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint] Saved: {filename}")
    
    return filepath


def load_checkpoint(filepath: str, generator, discriminator, 
                    g_optimizer=None, d_optimizer=None, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        generator: Generator model
        discriminator: Discriminator model
        g_optimizer: Generator optimizer (optional)
        d_optimizer: Discriminator optimizer (optional)
        device: Device to load to
    
    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if g_optimizer and 'g_optimizer_state_dict' in checkpoint:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    
    if d_optimizer and 'd_optimizer_state_dict' in checkpoint:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    print(f"[Checkpoint] Loaded: {os.path.basename(filepath)} (Epoch {checkpoint['epoch']})")
    
    return {
        'epoch': checkpoint['epoch'],
        'plant': checkpoint.get('plant', 'unknown'),
        'disease': checkpoint.get('disease', 'unknown'),
        'stage': checkpoint.get('stage', 1),
        'd_loss': checkpoint.get('d_loss', 0),
        'g_loss': checkpoint.get('g_loss', 0),
    }


def load_generator_only(filepath: str, generator, device='cuda'):
    """
    Load only the generator weights (for fine-tuning).
    
    Args:
        filepath: Path to checkpoint file
        generator: Generator model
        device: Device to load to
    """
    checkpoint = torch.load(filepath, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"[Checkpoint] Loaded Generator from: {os.path.basename(filepath)}")


def generate_sample_grid(generator, latent_dim: int, device: str, 
                         n_samples: int = 64, nrow: int = 8):
    """
    Generate a grid of sample images.
    
    Args:
        generator: Generator model
        latent_dim: Dimension of latent vector
        device: Device to use
        n_samples: Number of samples to generate
        nrow: Number of images per row
    
    Returns:
        Grid as numpy array (H, W, C) in [0, 255]
    """
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, 1, 1, device=device)
        fake_imgs = generator(z)
    
    generator.train()
    
    # Create grid
    grid = make_grid(fake_imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    # Convert to numpy
    grid = grid.cpu().permute(1, 2, 0).numpy()
    grid = (grid * 255).astype(np.uint8)
    
    return grid


def save_sample_grid(generator, latent_dim: int, device: str,
                     epoch: int, plant: str, disease: str, stage: int,
                     n_samples: int = 64, nrow: int = 8):
    """
    Generate and save a grid of sample images.
    
    Args:
        generator: Generator model
        latent_dim: Dimension of latent vector
        device: Device to use
        epoch: Current epoch
        plant: Plant name
        disease: Disease type
        stage: Training stage
        n_samples: Number of samples
        nrow: Images per row
    
    Returns:
        Path to saved image
    """
    grid = generate_sample_grid(generator, latent_dim, device, n_samples, nrow)
    
    # Create plant-specific folder
    stage_name = "healthy" if stage == 1 else disease
    plant_folder = os.path.join(SAMPLES_DIR, f"{plant}_{stage_name}")
    os.makedirs(plant_folder, exist_ok=True)
    
    # Save
    filename = f"epoch{epoch:03d}.png"
    filepath = os.path.join(plant_folder, filename)
    
    Image.fromarray(grid).save(filepath)
    print(f"[Samples] Saved: {plant}_{stage_name}/{filename}")
    
    return filepath


def save_single_images(generator, latent_dim: int, device: str,
                       plant: str, disease: str, n_images: int = 16,
                       output_dir: str = None):
    """
    Generate and save individual images.
    
    Args:
        generator: Generator model
        latent_dim: Dimension of latent vector
        device: Device to use
        plant: Plant name
        disease: Disease type
        n_images: Number of images to generate
        output_dir: Output directory (default: SAMPLES_DIR/plant_disease/generated)
    
    Returns:
        List of saved file paths
    """
    if output_dir is None:
        output_dir = os.path.join(SAMPLES_DIR, f"{plant}_{disease}", "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    generator.eval()
    saved_paths = []
    
    with torch.no_grad():
        for i in range(n_images):
            z = torch.randn(1, latent_dim, 1, 1, device=device)
            fake_img = generator(z)
            
            # Denormalize
            img_np = denormalize(fake_img[0])
            
            # Save
            filename = f"{plant}_{disease}_{i+1:04d}.png"
            filepath = os.path.join(output_dir, filename)
            Image.fromarray(img_np).save(filepath)
            saved_paths.append(filepath)
    
    generator.train()
    print(f"[Generated] {n_images} images saved to {output_dir}")
    
    return saved_paths


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, plant: str, disease: str, stage: int):
        self.plant = plant
        self.disease = disease
        self.stage = stage
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'd_real': [],
            'd_fake': [],
        }
        
        # Create log file
        stage_name = "healthy" if stage == 1 else disease
        self.log_file = os.path.join(
            LOGS_DIR, f"{plant}_{stage_name}_training.log"
        )
    
    def log(self, epoch: int, d_loss: float, g_loss: float, 
            d_real: float = 0, d_fake: float = 0):
        """Log training metrics for an epoch."""
        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss)
        self.history['d_real'].append(d_real)
        self.history['d_fake'].append(d_fake)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}, "
                   f"D(real)={d_real:.4f}, D(fake)={d_fake:.4f}\n")
    
    def plot_losses(self, save_path: str = None):
        """Plot training losses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['d_loss']) + 1)
        
        ax1.plot(epochs, self.history['d_loss'], label='D Loss')
        ax1.plot(epochs, self.history['g_loss'], label='G Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.history['d_real'], label='D(real)')
        ax2.plot(epochs, self.history['d_fake'], label='D(fake)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Probability')
        ax2.set_title('Discriminator Outputs')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            stage_name = "healthy" if self.stage == 1 else self.disease
            save_path = os.path.join(LOGS_DIR, f"{self.plant}_{stage_name}_losses.png")
        
        plt.savefig(save_path)
        plt.close()
        print(f"[Plot] Saved: {save_path}")


def find_latest_checkpoint(plant: str, disease: str = None, stage: int = None):
    """
    Find the latest checkpoint for a plant-disease combination.
    
    Args:
        plant: Plant name
        disease: Disease type (optional)
        stage: Training stage (optional)
    
    Returns:
        Path to latest checkpoint or None
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    # Build pattern
    if disease:
        pattern = f"G_{plant}_{disease}"
    elif stage == 1:
        pattern = f"G_{plant}_healthy"
    else:
        pattern = f"G_{plant}"
    
    # Find matching checkpoints
    checkpoints = []
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith(pattern) and f.endswith('.pth'):
            checkpoints.append(f)
    
    if not checkpoints:
        return None
    
    # Sort by epoch number and return latest
    checkpoints.sort()
    return os.path.join(CHECKPOINT_DIR, checkpoints[-1])


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test denormalize
    tensor = torch.randn(1, 3, 128, 128) * 0.5  # Values roughly in [-0.5, 0.5]
    result = denormalize(tensor)
    print(f"Denormalized shape: {result.shape}")
    print(f"Denormalized range: [{result.min()}, {result.max()}]")
    
    print("Utilities test passed!")
