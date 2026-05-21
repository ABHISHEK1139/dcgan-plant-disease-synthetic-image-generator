# DCGAN Training with Perceptual Loss for Sharper Images
# Adds VGG-based perceptual loss to standard GAN training

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LATENT_DIM, BATCH_SIZE, STAGE1_LR, BETA1, BETA2,
    LABEL_SMOOTHING_REAL, LABEL_SMOOTHING_FAKE, LABEL_FLIP_RATE,
    NOISE_STD_INITIAL, NOISE_DECAY, GRAD_CLIP_VALUE,
    SAMPLE_INTERVAL, CHECKPOINT_DIR, PLANT_DISEASE_MAP
)
from src.models import Generator, Discriminator, add_noise_to_inputs
from src.data_loader import create_dataloader
from src.utils import (
    save_checkpoint, load_checkpoint, load_generator_only,
    save_sample_grid, TrainingLogger, find_latest_checkpoint
)
from src.perceptual_loss import VGGPerceptualLoss


def get_labels(batch_size: int, real: bool, device: str, 
               smooth: bool = True, flip_rate: float = 0.0):
    """Generate labels with optional smoothing and random flipping."""
    if real:
        labels = torch.full((batch_size, 1), LABEL_SMOOTHING_REAL if smooth else 1.0, 
                           device=device)
    else:
        labels = torch.full((batch_size, 1), LABEL_SMOOTHING_FAKE if smooth else 0.0,
                           device=device)
    
    if flip_rate > 0:
        flip_mask = torch.rand(batch_size, 1, device=device) < flip_rate
        labels = torch.where(flip_mask, 1.0 - labels, labels)
    
    return labels


def train_epoch_sharp(generator, discriminator, dataloader, g_optimizer, d_optimizer,
                      criterion, perceptual_loss, device, epoch, 
                      noise_std: float = 0.0, perceptual_weight: float = 0.1):
    """
    Train for one epoch with perceptual loss for sharper images.
    
    The perceptual loss encourages the generator to produce images that
    match real images in VGG feature space, resulting in:
    - Sharper edges
    - Better leaf vein patterns
    - More realistic textures
    """
    generator.train()
    discriminator.train()
    
    total_d_loss = 0
    total_g_loss = 0
    total_perc_loss = 0
    total_d_real = 0
    total_d_fake = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for real_imgs in pbar:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # ==================== Train Discriminator ====================
        d_optimizer.zero_grad()
        
        real_labels = get_labels(batch_size, real=True, device=device, 
                                smooth=True, flip_rate=LABEL_FLIP_RATE)
        real_imgs_noisy = add_noise_to_inputs(real_imgs, noise_std)
        real_output = discriminator(real_imgs_noisy)
        d_loss_real = criterion(real_output, real_labels)
        
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = generator(z)
        fake_labels = get_labels(batch_size, real=False, device=device,
                                smooth=True, flip_rate=LABEL_FLIP_RATE)
        
        fake_imgs_noisy = add_noise_to_inputs(fake_imgs.detach(), noise_std)
        fake_output = discriminator(fake_imgs_noisy)
        d_loss_fake = criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_VALUE)
        d_optimizer.step()
        
        # ==================== Train Generator ====================
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = generator(z)
        
        real_labels_for_g = get_labels(batch_size, real=True, device=device,
                                       smooth=False, flip_rate=0.0)
        
        output = discriminator(fake_imgs)
        g_loss_adv = criterion(output, real_labels_for_g)
        
        # Perceptual loss for sharpness (compare with random real images)
        perc_loss = perceptual_loss(fake_imgs, real_imgs)
        
        # Combined loss
        g_loss = g_loss_adv + perceptual_weight * perc_loss
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_VALUE)
        g_optimizer.step()
        
        # Track metrics
        total_d_loss += d_loss.item()
        total_g_loss += g_loss_adv.item()
        total_perc_loss += perc_loss.item()
        total_d_real += real_output.mean().item()
        total_d_fake += fake_output.mean().item()
        num_batches += 1
        
        pbar.set_postfix({
            'D': f'{d_loss.item():.3f}',
            'G': f'{g_loss_adv.item():.3f}',
            'Perc': f'{perc_loss.item():.3f}',
            'D(x)': f'{real_output.mean().item():.2f}'
        })
    
    return (
        total_d_loss / num_batches,
        total_g_loss / num_batches,
        total_perc_loss / num_batches,
        total_d_real / num_batches,
        total_d_fake / num_batches
    )


def train_sharp(plant: str, disease: str, device: str = 'cuda',
                epochs: int = 150, lr: float = 0.0002,
                perceptual_weight: float = 0.1,
                resume_checkpoint: str = None):
    """
    Train with perceptual loss for sharper images.
    
    Args:
        plant: Plant name
        disease: Disease type
        device: Device to use
        epochs: Number of epochs
        lr: Learning rate
        perceptual_weight: Weight for perceptual loss (0.05-0.2 recommended)
        resume_checkpoint: Checkpoint to resume from
    """
    print(f"\n{'='*60}")
    print(f"SHARP TRAINING: {plant} - {disease}")
    print(f"Epochs: {epochs}, LR: {lr}, Perceptual Weight: {perceptual_weight}")
    print(f"{'='*60}\n")
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize perceptual loss
    perceptual_loss = VGGPerceptualLoss(device=device)
    print("VGG Perceptual Loss initialized for sharper images")
    
    # Setup optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(BETA1, BETA2))
    
    criterion = nn.BCELoss()
    
    start_epoch = 1
    
    # Load checkpoint if specified
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        info = load_checkpoint(resume_checkpoint, generator, discriminator,
                              g_optimizer, d_optimizer, device)
        start_epoch = info['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Logger
    logger = TrainingLogger(plant, disease, stage=2)
    
    noise_std = NOISE_STD_INITIAL
    
    for epoch in range(start_epoch, epochs + 1):
        d_loss, g_loss, perc_loss, d_real, d_fake = train_epoch_sharp(
            generator, discriminator, dataloader=create_dataloader(plant, disease, batch_size=BATCH_SIZE),
            g_optimizer=g_optimizer, d_optimizer=d_optimizer,
            criterion=criterion, perceptual_loss=perceptual_loss,
            device=device, epoch=epoch, noise_std=noise_std,
            perceptual_weight=perceptual_weight
        )
        
        logger.log(epoch, d_loss, g_loss, d_real, d_fake)
        
        print(f"\nEpoch {epoch}/{epochs} - D: {d_loss:.4f}, G: {g_loss:.4f}, "
              f"Perc: {perc_loss:.4f}, D(x): {d_real:.3f}")
        
        noise_std *= NOISE_DECAY
        
        if epoch % SAMPLE_INTERVAL == 0:
            save_sample_grid(generator, LATENT_DIM, device, epoch, 
                           plant, disease, stage=2, n_samples=64)
        
        if epoch % 10 == 0 or epoch == epochs:
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                           epoch, plant, disease, stage=2, d_loss=d_loss, g_loss=g_loss)
    
    save_sample_grid(generator, LATENT_DIM, device, epochs, 
                    plant, disease, stage=2, n_samples=64)
    logger.plot_losses()
    
    print(f"\nSharp training complete for {plant} - {disease}!")


def main():
    parser = argparse.ArgumentParser(description='DCGAN Sharp Training with Perceptual Loss')
    
    parser.add_argument('--plant', type=str, required=True,
                       help='Plant name')
    parser.add_argument('--disease', type=str, required=True,
                       help='Disease type')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                       help='Weight for perceptual loss (0.05-0.2 recommended)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    train_sharp(
        plant=args.plant,
        disease=args.disease,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        perceptual_weight=args.perceptual_weight,
        resume_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()
