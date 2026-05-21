# DCGAN Crop Leaf Disease Synthesis - Evaluation
# Visual inspection, metrics, and memorization checks

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LATENT_DIM, SAMPLES_DIR, CHECKPOINT_DIR
from src.models import Generator
from src.data_loader import create_dataloader
from src.utils import denormalize, load_generator_only


def visual_comparison(generator, dataloader, device: str,
                      n_samples: int = 8, save_path: str = None):
    """
    Create a side-by-side comparison of real vs generated images.
    
    Args:
        generator: Generator model
        dataloader: DataLoader for real images
        device: Device to use
        n_samples: Number of samples to compare
        save_path: Path to save the comparison image
    """
    generator.eval()
    
    # Get real images
    real_batch = next(iter(dataloader))[:n_samples].to(device)
    
    # Generate fake images
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, 1, 1, device=device)
        fake_batch = generator(z)
    
    # Create comparison grid
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # Real
        real_img = denormalize(real_batch[i])
        axes[0, i].imshow(real_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Real', fontsize=12)
        
        # Fake
        fake_img = denormalize(fake_batch[i])
        axes[1, i].imshow(fake_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Generated', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Comparison] Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
    generator.train()


def generate_samples(checkpoint_path: str, plant: str, disease: str,
                     n_samples: int = 16, device: str = 'cuda',
                     output_dir: str = None):
    """
    Generate sample images from a trained generator.
    
    Args:
        checkpoint_path: Path to generator checkpoint
        plant: Plant name
        disease: Disease type
        n_samples: Number of samples to generate
        device: Device to use
        output_dir: Output directory for samples
    
    Returns:
        List of generated image arrays
    """
    # Load generator
    generator = Generator().to(device)
    load_generator_only(checkpoint_path, generator, device)
    generator.eval()
    
    # Generate
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, LATENT_DIM, 1, 1, device=device)
            fake_img = generator(z)
            img_np = denormalize(fake_img[0])
            samples.append(img_np)
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(samples):
            filepath = os.path.join(output_dir, f"{plant}_{disease}_{i+1:03d}.png")
            Image.fromarray(img).save(filepath)
        print(f"[Generate] Saved {n_samples} images to {output_dir}")
    
    return samples


def extract_features(images: torch.Tensor, model='simple'):
    """
    Extract features from images for memorization check.
    Uses average pooling as a simple feature extractor.
    
    Args:
        images: Batch of images (B, C, H, W)
        model: Feature extractor type ('simple')
    
    Returns:
        Feature vectors (B, feature_dim)
    """
    if model == 'simple':
        # Simple: flatten and use as features
        # Or use global average pooling at multiple scales
        features = []
        
        # Original scale
        feat1 = images.mean(dim=[2, 3])  # (B, C)
        features.append(feat1)
        
        # 2x2 pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (2, 2))
        feat2 = pooled.view(pooled.size(0), -1)  # (B, C*4)
        features.append(feat2)
        
        # 4x4 pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (4, 4))
        feat3 = pooled.view(pooled.size(0), -1)  # (B, C*16)
        features.append(feat3)
        
        # 8x8 pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (8, 8))
        feat4 = pooled.view(pooled.size(0), -1)  # (B, C*64)
        features.append(feat4)
        
        return torch.cat(features, dim=1)
    
    else:
        raise ValueError(f"Unknown feature model: {model}")


def nearest_neighbor_check(generator, dataloader, device: str,
                           n_generated: int = 100, k: int = 1):
    """
    Check for memorization by finding nearest neighbors in feature space.
    
    If generated images are very close to training images (low distance),
    it may indicate memorization.
    
    Args:
        generator: Generator model
        dataloader: DataLoader for real training images
        device: Device to use
        n_generated: Number of generated images to check
        k: Number of nearest neighbors
    
    Returns:
        Dictionary with distances statistics
    """
    generator.eval()
    
    # Collect real image features
    print("Extracting real image features...")
    real_features = []
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            feat = extract_features(batch)
        real_features.append(feat.cpu())
    real_features = torch.cat(real_features, dim=0)
    print(f"Real features shape: {real_features.shape}")
    
    # Generate fake images and extract features
    print("Generating and extracting fake image features...")
    fake_features = []
    with torch.no_grad():
        for _ in range(0, n_generated, 16):
            batch_size = min(16, n_generated - len(fake_features) * 16)
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)
            feat = extract_features(fake_imgs)
            fake_features.append(feat.cpu())
    fake_features = torch.cat(fake_features, dim=0)[:n_generated]
    print(f"Fake features shape: {fake_features.shape}")
    
    # Compute pairwise distances
    print("Computing nearest neighbor distances...")
    min_distances = []
    
    for i in range(n_generated):
        fake_feat = fake_features[i:i+1]  # (1, D)
        distances = torch.cdist(fake_feat, real_features)[0]  # (N,)
        
        # Get k-nearest distances
        knn_distances = torch.topk(distances, k, largest=False)[0]
        min_distances.append(knn_distances[0].item())
    
    # Statistics
    min_distances = np.array(min_distances)
    stats = {
        'mean_distance': float(np.mean(min_distances)),
        'std_distance': float(np.std(min_distances)),
        'min_distance': float(np.min(min_distances)),
        'max_distance': float(np.max(min_distances)),
        'median_distance': float(np.median(min_distances)),
    }
    
    print("\nNearest Neighbor Statistics:")
    print(f"  Mean distance: {stats['mean_distance']:.4f}")
    print(f"  Std distance: {stats['std_distance']:.4f}")
    print(f"  Min distance: {stats['min_distance']:.4f}")
    print(f"  Max distance: {stats['max_distance']:.4f}")
    print(f"  Median distance: {stats['median_distance']:.4f}")
    
    # Interpretation
    if stats['min_distance'] < 0.1:
        print("\n⚠️ WARNING: Very low minimum distance detected!")
        print("   This may indicate memorization of training images.")
    else:
        print("\n✓ Distance check passed - no obvious memorization detected.")
    
    generator.train()
    return stats


def plot_training_progress(samples_dir: str, plant: str, disease: str,
                           epochs_to_show: list = None, save_path: str = None):
    """
    Create a grid showing training progress across epochs.
    
    Args:
        samples_dir: Directory containing sample grids
        plant: Plant name
        disease: Disease type
        epochs_to_show: List of epochs to display
        save_path: Path to save the figure
    """
    folder = os.path.join(samples_dir, f"{plant}_{disease}")
    
    if not os.path.exists(folder):
        print(f"No samples found in {folder}")
        return
    
    # Find available epochs
    available = []
    for f in os.listdir(folder):
        if f.startswith('epoch') and f.endswith('.png'):
            epoch = int(f.replace('epoch', '').replace('.png', ''))
            available.append(epoch)
    available.sort()
    
    if not available:
        print("No epoch samples found")
        return
    
    # Select epochs to show
    if epochs_to_show is None:
        # Show first, middle, and last
        if len(available) >= 5:
            epochs_to_show = [available[0], available[len(available)//4], 
                            available[len(available)//2], available[3*len(available)//4],
                            available[-1]]
        else:
            epochs_to_show = available
    
    # Load and display
    n_epochs = len(epochs_to_show)
    fig, axes = plt.subplots(1, n_epochs, figsize=(4*n_epochs, 4))
    
    if n_epochs == 1:
        axes = [axes]
    
    for i, epoch in enumerate(epochs_to_show):
        img_path = os.path.join(folder, f"epoch{epoch:03d}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Epoch {epoch}")
        axes[i].axis('off')
    
    plt.suptitle(f"Training Progress: {plant} - {disease}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Progress] Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DCGAN Evaluation')
    parser.add_argument('--plant', type=str, required=True)
    parser.add_argument('--disease', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--action', type=str, default='compare',
                       choices=['compare', 'generate', 'memorization', 'progress'])
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    if args.action == 'generate' and args.checkpoint:
        generate_samples(args.checkpoint, args.plant, args.disease,
                        n_samples=args.n, device=args.device)
    
    elif args.action == 'progress':
        plot_training_progress(SAMPLES_DIR, args.plant, args.disease)
    
    else:
        print(f"Action '{args.action}' requires additional setup")
