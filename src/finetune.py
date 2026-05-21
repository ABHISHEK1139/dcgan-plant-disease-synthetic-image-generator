# DCGAN Fine-tuning with Perceptual Loss
# For sharper, more detailed leaf images

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
    LABEL_SMOOTHING_REAL, LABEL_SMOOTHING_FAKE,
    GRAD_CLIP_VALUE, SAMPLE_INTERVAL, CHECKPOINT_DIR, PLANT_DISEASE_MAP
)
from src.models import Generator, Discriminator
from src.data_loader import create_dataloader
from src.utils import (
    save_checkpoint, load_checkpoint, save_sample_grid, 
    TrainingLogger, find_latest_checkpoint
)
from src.perceptual_loss import VGGPerceptualLoss


def get_labels(batch_size: int, real: bool, device: str):
    """Generate labels with smoothing."""
    if real:
        return torch.full((batch_size, 1), LABEL_SMOOTHING_REAL, device=device)
    else:
        return torch.full((batch_size, 1), LABEL_SMOOTHING_FAKE, device=device)


def finetune_with_perceptual(
    plant: str, 
    disease: str,
    checkpoint_path: str,
    epochs: int = 50,
    lr: float = 0.00005,  # Very low LR for fine-tuning
    perceptual_weight: float = 0.1,  # Weight for perceptual loss
    device: str = 'cuda'
):
    """
    Fine-tune a trained model with perceptual loss for sharper details.
    
    Args:
        plant: Plant name
        disease: Disease type (or 'healthy')
        checkpoint_path: Path to trained checkpoint
        epochs: Fine-tuning epochs
        lr: Learning rate (should be very low)
        perceptual_weight: Weight for perceptual loss term
        device: Device to use
    """
    print(f"\n{'='*60}")
    print(f"Fine-tuning with Perceptual Loss")
    print(f"Plant: {plant}, Disease: {disease}")
    print(f"Epochs: {epochs}, LR: {lr}, Perceptual Weight: {perceptual_weight}")
    print(f"{'='*60}\n")
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Load checkpoint
    info = load_checkpoint(checkpoint_path, generator, discriminator, device=device)
    start_epoch = info['epoch'] + 1
    print(f"Loaded checkpoint from epoch {info['epoch']}")
    
    # Initialize perceptual loss
    print("Loading VGG for perceptual loss...")
    perceptual_loss_fn = VGGPerceptualLoss(device=device)
    
    # Setup optimizers with low LR
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(BETA1, BETA2))
    
    # Loss functions
    bce_loss = nn.BCELoss()
    
    # Load data
    dataloader = create_dataloader(plant, disease, batch_size=BATCH_SIZE)
    
    # Logger
    stage = 1 if disease == "healthy" else 2
    logger = TrainingLogger(plant, f"{disease}_finetune", stage)
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        generator.train()
        discriminator.train()
        
        total_d_loss = 0
        total_g_loss = 0
        total_p_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for real_imgs in pbar:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # ==================== Train Discriminator ====================
            d_optimizer.zero_grad()
            
            # Real images
            real_labels = get_labels(batch_size, real=True, device=device)
            real_output = discriminator(real_imgs)
            d_loss_real = bce_loss(real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)
            fake_labels = get_labels(batch_size, real=False, device=device)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = bce_loss(fake_output, fake_labels)
            
            # Backward D
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_VALUE)
            d_optimizer.step()
            
            # ==================== Train Generator ====================
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)
            
            # Adversarial loss (fool D)
            real_labels_for_g = get_labels(batch_size, real=True, device=device)
            output = discriminator(fake_imgs)
            g_loss_adv = bce_loss(output, real_labels_for_g)
            
            # Perceptual loss (match VGG features)
            p_loss = perceptual_loss_fn(fake_imgs, real_imgs)
            
            # Combined G loss
            g_loss = g_loss_adv + perceptual_weight * p_loss
            
            # Backward G
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_VALUE)
            g_optimizer.step()
            
            # Track metrics
            total_d_loss += d_loss.item()
            total_g_loss += g_loss_adv.item()
            total_p_loss += p_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_loss_adv.item():.3f}',
                'P': f'{p_loss.item():.3f}'
            })
        
        # Epoch stats
        avg_d = total_d_loss / num_batches
        avg_g = total_g_loss / num_batches
        avg_p = total_p_loss / num_batches
        
        print(f"\nEpoch {epoch} - D: {avg_d:.4f}, G: {avg_g:.4f}, Perceptual: {avg_p:.4f}")
        logger.log(epoch, avg_d, avg_g)
        
        # Save samples every 5 epochs
        if epoch % SAMPLE_INTERVAL == 0 or epoch == start_epoch + epochs - 1:
            save_sample_grid(generator, LATENT_DIM, device, epoch, 
                           plant, f"{disease}_finetune", stage, n_samples=64)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == start_epoch + epochs - 1:
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                          epoch, plant, f"{disease}_finetune", stage, avg_d, avg_g)
    
    print(f"\nFine-tuning complete!")
    print(f"Final model saved to: checkpoints/")
    print(f"Samples saved to: samples/{plant}_{disease}_finetune/")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DCGAN with Perceptual Loss')
    
    parser.add_argument('--plant', type=str, required=True)
    parser.add_argument('--disease', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to fine-tune (default: latest)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--perceptual-weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(args.plant, args.disease)
        if not checkpoint_path:
            print(f"No checkpoint found for {args.plant}/{args.disease}")
            return
    
    # Run fine-tuning
    finetune_with_perceptual(
        plant=args.plant,
        disease=args.disease,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        lr=args.lr,
        perceptual_weight=args.perceptual_weight,
        device=args.device
    )


if __name__ == "__main__":
    main()
