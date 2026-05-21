# DCGAN Crop Leaf Disease Synthesis - Optimized Model Architecture
# Fast training, balanced D/G, good quality for RTX 3050

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LATENT_DIM, IMG_CHANNELS


def weights_init(m):
    """Initialize weights from a normal distribution with mean 0 and std 0.02."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Sharp DCGAN Generator for 128x128 images.
    
    Uses Upsample + Conv2d instead of ConvTranspose2d to avoid
    checkerboard artifacts and produce sharper images.
    
    Architecture:
        Input: (batch, 100, 1, 1) latent vector
        → FC to 512x4x4
        → 5 Upsample+Conv blocks with BatchNorm + ReLU
        → Output: (batch, 3, 128, 128) image in [-1, 1]
    
    Path: 4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128
    """
    
    def __init__(self, latent_dim: int = LATENT_DIM, img_channels: int = IMG_CHANNELS):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Initial projection from latent to 4x4 feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(True)
        )
        
        def upsample_block(in_ch, out_ch):
            """Upsample + Conv block - no checkerboard artifacts"""
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )
        
        self.main = nn.Sequential(
            # 4x4 → 8x8
            upsample_block(512, 256),
            
            # 8x8 → 16x16
            upsample_block(256, 128),
            
            # 16x16 → 32x32
            upsample_block(128, 64),
            
            # 32x32 → 64x64
            upsample_block(64, 64),  # Keep 64 channels for more capacity
            
            # 64x64 → 128x128
            upsample_block(64, 32),
            
            # Final conv to RGB
            nn.Conv2d(32, img_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        self.apply(weights_init)
    
    def forward(self, z):
        # Handle both (batch, latent) and (batch, latent, 1, 1) inputs
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        return self.main(x)


class LegacyGenerator(nn.Module):
    """
    Legacy DCGAN Generator for 128x128 images.
    Uses ConvTranspose2d (standard implementation).
    Kept for backward compatibility with older checkpoints.
    """
    def __init__(self, latent_dim: int = LATENT_DIM, img_channels: int = IMG_CHANNELS):
        super(LegacyGenerator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (512) x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (256) x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (128) x 16 x 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. (32) x 64 x 64
            
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 128 x 128
        )
        self.apply(weights_init)

    def forward(self, input):
        if input.dim() == 2:
             input = input.unsqueeze(2).unsqueeze(3)
        return self.main(input)


class Discriminator(nn.Module):
    """
    Optimized DCGAN Discriminator for 128x128 images.
    
    Key improvements:
    - NO spectral norm (was making D too weak)
    - Moderate dropout (0.25) for regularization
    - Strong first layer to catch low-level features
    
    Architecture:
        Input: (batch, 3, 128, 128) image
        → 6 Conv blocks with LeakyReLU + BatchNorm + Dropout
        → Output: (batch, 1) probability
    """
    
    def __init__(self, img_channels: int = IMG_CHANNELS):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 128x128 → 64x64 (no BatchNorm in first layer)
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 → 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 32x32 → 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 16x16 → 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 8x8 → 4x4
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 → 1x1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        self.apply(weights_init)
    
    def forward(self, img):
        return self.main(img).view(-1, 1)


def add_noise_to_inputs(images: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to discriminator inputs for stability."""
    if std > 0:
        noise = torch.randn_like(images) * std
        return images + noise
    return images


if __name__ == "__main__":
    print("Testing Optimized Generator and Discriminator...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test Generator
    G = Generator().to(device)
    z = torch.randn(4, LATENT_DIM, 1, 1, device=device)
    fake_imgs = G(z)
    print(f"Generator output shape: {fake_imgs.shape}")
    
    # Test Discriminator
    D = Discriminator().to(device)
    output = D(fake_imgs)
    print(f"Discriminator output shape: {output.shape}")
    
    # Count parameters
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    print("\nOptimized model test passed!")
