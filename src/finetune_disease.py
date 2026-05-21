# DCGAN Fine-Tuning for Disease Synthesis
# Based on best practices: Load BOTH G+D, same LR, few epochs

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LATENT_DIM, BATCH_SIZE, BETA1, BETA2,
    LABEL_SMOOTHING_REAL, LABEL_SMOOTHING_FAKE, LABEL_FLIP_RATE,
    NOISE_STD_INITIAL, NOISE_DECAY, GRAD_CLIP_VALUE,
    SAMPLE_INTERVAL, CHECKPOINT_DIR
)
from src.models import Generator, Discriminator, add_noise_to_inputs
from src.data_loader import create_dataloader
from src.utils import save_checkpoint, save_sample_grid, TrainingLogger


# Best practice settings for DCGAN fine-tuning
DEFAULT_LR = 1e-5        # Same LR for G and D (20-40x smaller than base training)
DEFAULT_EPOCHS = 20      # Short fine-tuning (10-30 epochs with early stopping)


def get_labels(batch_size, real, device, smooth=True, flip_rate=0.0):
    if real:
        labels = torch.full((batch_size, 1), LABEL_SMOOTHING_REAL if smooth else 1.0, device=device)
    else:
        labels = torch.full((batch_size, 1), LABEL_SMOOTHING_FAKE if smooth else 0.0, device=device)
    if flip_rate > 0:
        flip_mask = torch.rand(batch_size, 1, device=device) < flip_rate
        labels = torch.where(flip_mask, 1.0 - labels, labels)
    return labels


def freeze_early_layers(generator, num_blocks: int = 2):
    """Freeze early generator blocks to preserve low-level features."""
    if num_blocks <= 0:
        return
    
    layers = list(generator.main.children())
    layers_per_block = 3
    layers_to_freeze = min(num_blocks * layers_per_block, len(layers) - 3)
    
    frozen_count = 0
    for i, layer in enumerate(layers[:layers_to_freeze]):
        for param in layer.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
    
    trainable_count = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Frozen {num_blocks} early G blocks ({frozen_count:,} params)")
    print(f"Trainable G params: {trainable_count:,}")


def finetune_disease(plant: str, disease: str, base_checkpoint: str,
                     epochs: int = DEFAULT_EPOCHS, 
                     lr: float = DEFAULT_LR,
                     freeze_blocks: int = 1,
                     device: str = 'cuda'):
    """
    Fine-tune from healthy checkpoint to learn disease patterns.
    
    Key best practices:
    - Load BOTH Generator AND Discriminator weights
    - Same LR for G and D (20-40x smaller than base)
    - Short training: 10-30 epochs with early stopping
    
    Args:
        plant: Plant name
        disease: Target disease
        base_checkpoint: Path to healthy checkpoint (contains G+D)
        epochs: Number of epochs (default: 20)
        lr: Learning rate (default: 1e-5, same for G and D)
        freeze_blocks: Freeze N early G blocks (default: 1)
        device: cuda or cpu
    """
    print(f"\n{'='*60}")
    print(f"DCGAN FINE-TUNING: {plant} - {disease}")
    print(f"Base: {base_checkpoint}")
    print(f"LR: {lr:.0e} (same for G and D)")
    print(f"Epochs: {epochs}")
    print(f"Freeze blocks: {freeze_blocks}")
    print(f"{'='*60}\n")
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Load checkpoint
    if not os.path.exists(base_checkpoint):
        print(f"ERROR: Checkpoint not found: {base_checkpoint}")
        return
    
    print(f"Loading healthy checkpoint...")
    checkpoint = torch.load(base_checkpoint, map_location=device, weights_only=False)
    
    # Load BOTH Generator AND Discriminator weights
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print("✓ Loaded Generator weights")
        
        if 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            print("✓ Loaded Discriminator weights")
        else:
            print("⚠ No Discriminator weights - using fresh D")
    else:
        # Raw state dict (generator only)
        generator.load_state_dict(checkpoint)
        print("✓ Loaded Generator (raw state dict)")
        print("⚠ Using fresh Discriminator")
    
    # Freeze early layers
    if freeze_blocks > 0:
        freeze_early_layers(generator, freeze_blocks)
    
    # IMPORTANT: Same LR for both G and D
    g_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=lr, betas=(BETA1, BETA2)
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(BETA1, BETA2))
    
    criterion = nn.BCELoss()
    
    # Load disease dataset
    dataloader = create_dataloader(plant, disease, batch_size=BATCH_SIZE)
    print(f"Loaded {len(dataloader.dataset)} disease images\n")
    
    logger = TrainingLogger(plant, disease, stage=2)
    noise_std = NOISE_STD_INITIAL * 0.5  # Lower noise for fine-tuning
    
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        total_d_loss = 0
        total_g_loss = 0
        total_d_real = 0
        total_d_fake = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for real_imgs in pbar:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            real_labels = get_labels(batch_size, True, device, True, LABEL_FLIP_RATE)
            real_imgs_noisy = add_noise_to_inputs(real_imgs, noise_std)
            real_output = discriminator(real_imgs_noisy)
            d_loss_real = criterion(real_output, real_labels)
            
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)
            fake_labels = get_labels(batch_size, False, device, True, LABEL_FLIP_RATE)
            fake_imgs_noisy = add_noise_to_inputs(fake_imgs.detach(), noise_std)
            fake_output = discriminator(fake_imgs_noisy)
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_VALUE)
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)
            real_labels_g = get_labels(batch_size, True, device, False, 0.0)
            output = discriminator(fake_imgs)
            g_loss = criterion(output, real_labels_g)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_VALUE)
            g_optimizer.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            total_d_real += real_output.mean().item()
            total_d_fake += fake_output.mean().item()
            num_batches += 1
            
            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_loss.item():.3f}',
                'D(x)': f'{real_output.mean().item():.2f}',
                'D(G)': f'{fake_output.mean().item():.2f}'
            })
        
        avg_d = total_d_loss / num_batches
        avg_g = total_g_loss / num_batches
        avg_real = total_d_real / num_batches
        avg_fake = total_d_fake / num_batches
        
        logger.log(epoch, avg_d, avg_g, avg_real, avg_fake)
        print(f"\nEpoch {epoch}: D={avg_d:.4f}, G={avg_g:.4f}, D(x)={avg_real:.3f}, D(G)={avg_fake:.3f}")
        
        # Save samples every epoch for short training
        save_sample_grid(generator, LATENT_DIM, device, epoch, plant, disease, stage=2, n_samples=64)
        
        # Save checkpoint
        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                           epoch, plant, disease, stage=2, d_loss=avg_d, g_loss=avg_g)
        
        noise_std *= NOISE_DECAY
    
    # Final save
    save_sample_grid(generator, LATENT_DIM, device, epoch, plant, disease, stage=2, n_samples=64)
    logger.plot_losses()
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING COMPLETE: {plant} - {disease}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='DCGAN Fine-tuning for Disease Synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Best practice settings:
  LR: 1e-5 (same for G and D, 20-40x smaller than base)
  Epochs: 10-30 (with early stopping)
  Load BOTH Generator AND Discriminator from healthy checkpoint

Example:
  python -m src.finetune_disease --plant Tomato --disease Late_blight \\
    --base checkpoints/G_Tomato_healthy_epoch330.pth \\
    --lr 1e-5 --epochs 20 --freeze-blocks 1
        """
    )
    parser.add_argument('--plant', type=str, required=True)
    parser.add_argument('--disease', type=str, required=True)
    parser.add_argument('--base', type=str, required=True, help='Healthy checkpoint path')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='LR for both G and D')
    parser.add_argument('--freeze-blocks', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    finetune_disease(
        plant=args.plant,
        disease=args.disease,
        base_checkpoint=args.base,
        epochs=args.epochs,
        lr=args.lr,
        freeze_blocks=args.freeze_blocks,
        device=args.device
    )


if __name__ == "__main__":
    main()
