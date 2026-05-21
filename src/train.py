# DCGAN Crop Leaf Disease Synthesis - Training Script
# Two-Stage Training: Stage 1 (Healthy Anatomy) → Stage 2 (Disease Fine-tuning)

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LATENT_DIM, BATCH_SIZE, STAGE1_EPOCHS, STAGE2_EPOCHS,
    STAGE1_LR, BETA1, BETA2,
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

# Perceptual loss weight - keep LOW to prevent mode collapse
PERCEPTUAL_WEIGHT = 0.001


def get_labels(batch_size: int, real: bool, device: str, 
               smooth: bool = True, flip_rate: float = 0.0):
    """
    Generate labels with optional smoothing and random flipping.
    
    Args:
        batch_size: Number of labels
        real: True for real labels, False for fake
        device: Device to create tensor on
        smooth: Apply label smoothing
        flip_rate: Probability of flipping labels
    
    Returns:
        Label tensor
    """
    if real:
        labels = torch.full((batch_size, 1), LABEL_SMOOTHING_REAL if smooth else 1.0, 
                           device=device)
    else:
        labels = torch.full((batch_size, 1), LABEL_SMOOTHING_FAKE if smooth else 0.0,
                           device=device)
    
    # Random label flipping for training stability
    if flip_rate > 0:
        flip_mask = torch.rand(batch_size, 1, device=device) < flip_rate
        labels = torch.where(flip_mask, 1.0 - labels, labels)
    
    return labels


def train_epoch(generator, discriminator, dataloader, 
                criterion, g_optimizer, d_optimizer,
                device: str, epoch: int, noise_std: float = 0.0,
                perceptual_loss_fn=None, real_batch_cache=None):
    """
    Train for one epoch.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: DataLoader for real images
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        criterion: Loss function (BCELoss)
        device: Device to use
        epoch: Current epoch number
        noise_std: Noise to add to discriminator inputs
    
    Returns:
        (avg_d_loss, avg_g_loss, avg_d_real, avg_d_fake)
    """
    generator.train()
    discriminator.train()
    
    total_d_loss = 0
    total_g_loss = 0
    total_d_real = 0
    total_d_fake = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for real_imgs in pbar:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # ==================== Train Discriminator ====================
        d_optimizer.zero_grad()
        
        # Real images
        real_labels = get_labels(batch_size, real=True, device=device, 
                                smooth=True, flip_rate=LABEL_FLIP_RATE)
        
        # Add noise to real images
        real_imgs_noisy = add_noise_to_inputs(real_imgs, noise_std)
        real_output = discriminator(real_imgs_noisy)
        d_loss_real = criterion(real_output, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = generator(z)
        fake_labels = get_labels(batch_size, real=False, device=device,
                                smooth=True, flip_rate=LABEL_FLIP_RATE)
        
        # Add noise to fake images
        fake_imgs_noisy = add_noise_to_inputs(fake_imgs.detach(), noise_std)
        fake_output = discriminator(fake_imgs_noisy)
        d_loss_fake = criterion(fake_output, fake_labels)
        
        # Backward and optimize D
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_VALUE)
        d_optimizer.step()
        
        # ==================== Train Generator ====================
        g_optimizer.zero_grad()
        
        # Generate fake images and try to fool D
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = generator(z)
        
        # Generator wants D to output "real" for fake images
        real_labels_for_g = get_labels(batch_size, real=True, device=device,
                                       smooth=False, flip_rate=0.0)  # No smoothing for G
        
        output = discriminator(fake_imgs)
        g_adv_loss = criterion(output, real_labels_for_g)
        
        # Add perceptual loss for better detail
        g_loss = g_adv_loss
        if perceptual_loss_fn is not None and real_batch_cache is not None:
            p_loss = perceptual_loss_fn(fake_imgs, real_batch_cache)
            g_loss = g_adv_loss + PERCEPTUAL_WEIGHT * p_loss
        
        # Backward and optimize G
        g_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_VALUE)
        g_optimizer.step()
        
        # Track metrics
        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        total_d_real += real_output.mean().item()
        total_d_fake += fake_output.mean().item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'D_loss': f'{d_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}',
            'D(x)': f'{real_output.mean().item():.3f}',
            'D(G(z))': f'{fake_output.mean().item():.3f}'
        })
    
    # Return averages
    return (
        total_d_loss / num_batches,
        total_g_loss / num_batches,
        total_d_real / num_batches,
        total_d_fake / num_batches
    )


def train_stage(generator, discriminator, dataloader, device: str,
                plant: str, disease: str, stage: int,
                epochs: int, lr: float, resume_from: str = None):
    """
    Train for a complete stage (Stage 1 or Stage 2).
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: DataLoader for training images
        device: Device to use
        plant: Plant name
        disease: Disease type
        stage: Training stage (1 or 2)
        epochs: Number of epochs
        lr: Learning rate
        resume_from: Optional checkpoint to resume from
    """
    print(f"\n{'='*60}")
    print(f"Stage {stage}: Training on {plant} - {disease}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}\n")
    
    # Setup optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(BETA1, BETA2))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Starting epoch
    start_epoch = 1
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        info = load_checkpoint(resume_from, generator, discriminator,
                              g_optimizer, d_optimizer, device)
        start_epoch = info['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Initialize logger
    logger = TrainingLogger(plant, disease, stage)
    
    # Initialize perceptual loss for better detail (only for larger datasets)
    dataset_size = len(dataloader.dataset)
    # Perceptual loss DISABLED for faster training
    perceptual_loss_fn = None
    print(f"[Perceptual Loss] DISABLED - dataset size: {dataset_size}")
    
    # Noise decay
    noise_std = NOISE_STD_INITIAL
    
    # Cache for real images (for perceptual loss comparison)
    real_batch_cache = None
    
    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # Get a batch of real images for perceptual loss comparison
        for real_batch in dataloader:
            real_batch_cache = real_batch.to(device)
            break
        
        # Train one epoch
        d_loss, g_loss, d_real, d_fake = train_epoch(
            generator, discriminator, dataloader,
            criterion, g_optimizer, d_optimizer,
            device, epoch, noise_std,
            perceptual_loss_fn, real_batch_cache
        )
        
        # Log metrics
        logger.log(epoch, d_loss, g_loss, d_real, d_fake)
        
        print(f"\nEpoch {epoch}/{epochs} - D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}, "
              f"D(x): {d_real:.3f}, D(G(z)): {d_fake:.3f}")
        
        # Decay noise
        noise_std *= NOISE_DECAY
        
        # Save sample grid every SAMPLE_INTERVAL epochs
        if epoch % SAMPLE_INTERVAL == 0:
            save_sample_grid(generator, LATENT_DIM, device, epoch, 
                           plant, disease, stage, n_samples=32)
        
        # Save checkpoint every 5 epochs (more frequent for small datasets)
        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                           epoch, plant, disease, stage, d_loss, g_loss)
    
    # Final sample and loss plot
    save_sample_grid(generator, LATENT_DIM, device, epochs, 
                    plant, disease, stage, n_samples=32)
    logger.plot_losses()
    
    print(f"\nStage {stage} training complete!")


def run_two_stage_training(plant: str, target_disease: str, 
                           device: str = 'cuda',
                           stage1_epochs: int = None,
                           stage2_epochs: int = None,
                           resume_stage: int = None,
                           resume_checkpoint: str = None):
    """
    Run the complete two-stage training pipeline.
    
    Stage 1: Train on healthy leaves to learn anatomy
    Stage 2: Fine-tune on diseased leaves to add lesions
    
    Args:
        plant: Plant name (e.g., "Tomato")
        target_disease: Target disease (e.g., "Late_blight")
        device: Device to use
        stage1_epochs: Override Stage 1 epochs
        stage2_epochs: Override Stage 2 epochs
        resume_stage: Resume from stage 1 or 2
        resume_checkpoint: Specific checkpoint to resume from
    """
    if stage1_epochs is None:
        stage1_epochs = STAGE1_EPOCHS
    if stage2_epochs is None:
        stage2_epochs = STAGE2_EPOCHS
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    print(f"Device: {device}")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # ==================== Stage 1: Healthy (Anatomy) ====================
    if resume_stage is None or resume_stage == 1:
        print("\n" + "="*60)
        print("STAGE 1: Learning Healthy Leaf Anatomy")
        print("="*60)
        
        # Load healthy dataset
        healthy_loader = create_dataloader(plant, "healthy", batch_size=BATCH_SIZE)
        
        # Find resume checkpoint for Stage 1
        stage1_resume = None
        if resume_stage == 1 and resume_checkpoint:
            stage1_resume = resume_checkpoint
        elif resume_stage == 1:
            stage1_resume = find_latest_checkpoint(plant, "healthy")
        
        # Train Stage 1
        train_stage(
            generator, discriminator, healthy_loader, device,
            plant, "healthy", stage=1,
            epochs=stage1_epochs, lr=STAGE1_LR,
            resume_from=stage1_resume
        )
    
    # ==================== Stage 2: Disease (Pathology) ====================
    print("\n" + "="*60)
    print(f"STAGE 2: Learning {target_disease} Pathology")
    print("="*60)
    
    # Load Stage 1 base model if starting Stage 2 fresh
    if resume_stage == 2:
        if resume_checkpoint:
            load_checkpoint(resume_checkpoint, generator, discriminator, device=device)
        else:
            # Load latest Stage 1 checkpoint
            stage1_checkpoint = find_latest_checkpoint(plant, "healthy")
            if stage1_checkpoint:
                load_generator_only(stage1_checkpoint, generator, device)
            else:
                print("Warning: No Stage 1 checkpoint found. Starting from scratch.")
    
    # Reinitialize discriminator for Stage 2 (fresh discrimination)
    if resume_stage != 2:
        discriminator = Discriminator().to(device)
    
    # Load disease dataset
    disease_loader = create_dataloader(plant, target_disease, batch_size=BATCH_SIZE)
    
    # Find resume checkpoint for Stage 2
    stage2_resume = None
    if resume_stage == 2 and resume_checkpoint:
        stage2_resume = resume_checkpoint
    
    # Train Stage 2
    train_stage(
        generator, discriminator, disease_loader, device,
        plant, target_disease, stage=2,
        epochs=stage2_epochs, lr=STAGE2_LR,
        resume_from=stage2_resume
    )
    
    print("\n" + "="*60)
    print("TWO-STAGE TRAINING COMPLETE!")
    print(f"Base model: {plant}_healthy")
    print(f"Disease model: {plant}_{target_disease}")
    print("="*60)


def train_single_class(plant: str, disease: str, 
                       device: str = 'cuda',
                       epochs: int = 100,
                       lr: float = 0.0002,
                       resume_checkpoint: str = None):
    """
    Train on a single class (healthy or disease) without two-stage.
    Useful for quick experiments or when you want to train directly.
    
    Args:
        plant: Plant name
        disease: Disease type ("healthy" or specific disease)
        device: Device to use
        epochs: Number of epochs
        lr: Learning rate
        resume_checkpoint: Checkpoint to resume from
    """
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    print(f"Training {plant} - {disease}")
    print(f"Device: {device}")
    
    # Load dataset
    dataloader = create_dataloader(plant, disease, batch_size=BATCH_SIZE)
    
    # Determine stage
    stage = 1 if disease == "healthy" else 2
    
    # Train
    train_stage(
        generator, discriminator, dataloader, device,
        plant, disease, stage=stage,
        epochs=epochs, lr=lr,
        resume_from=resume_checkpoint
    )


def main():
    parser = argparse.ArgumentParser(description='DCGAN Two-Stage Training')
    
    parser.add_argument('--plant', type=str, required=True,
                       help='Plant name (e.g., Tomato, Potato)')
    parser.add_argument('--disease', type=str, default=None,
                       help='Target disease for Stage 2 (e.g., Late_blight)')
    parser.add_argument('--mode', type=str, default='two-stage',
                       choices=['two-stage', 'single', 'healthy'],
                       help='Training mode')
    parser.add_argument('--stage1-epochs', type=int, default=None,
                       help='Override Stage 1 epochs')
    parser.add_argument('--stage2-epochs', type=int, default=None,
                       help='Override Stage 2 epochs')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Epochs for single-class training')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate for single-class training')
    parser.add_argument('--resume-stage', type=int, default=None,
                       choices=[1, 2], help='Resume from stage')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint path to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Validate plant
    if args.plant not in PLANT_DISEASE_MAP:
        print(f"Error: Unknown plant '{args.plant}'")
        print(f"Available plants: {list(PLANT_DISEASE_MAP.keys())}")
        return
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run training
    if args.mode == 'two-stage':
        if args.disease is None:
            print("Error: --disease required for two-stage training")
            return
        
        run_two_stage_training(
            plant=args.plant,
            target_disease=args.disease,
            device=args.device,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            resume_stage=args.resume_stage,
            resume_checkpoint=args.resume
        )
    
    elif args.mode == 'healthy':
        train_single_class(
            plant=args.plant,
            disease='healthy',
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            resume_checkpoint=args.resume
        )
    
    elif args.mode == 'single':
        if args.disease is None:
            print("Error: --disease required for single-class training")
            return
        
        train_single_class(
            plant=args.plant,
            disease=args.disease,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            resume_checkpoint=args.resume
        )


if __name__ == "__main__":
    main()
