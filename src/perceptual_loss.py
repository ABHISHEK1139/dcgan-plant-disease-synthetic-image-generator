# DCGAN Crop Leaf Disease Synthesis - Perceptual Loss
# VGG-based feature matching for sharper, more detailed images

import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    
    Compares generated and real images in VGG feature space,
    which captures textures and structural details better than pixel loss.
    
    This helps produce:
    - Sharper edges
    - More detailed vein patterns
    - Better textures
    """
    
    def __init__(self, layers: list = None, device: str = 'cuda'):
        super(VGGPerceptualLoss, self).__init__()
        
        # Use specific VGG layers for feature extraction
        # Earlier layers capture edges, later layers capture texture
        if layers is None:
            layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        
        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg = vgg.to(device).eval()
        
        # Freeze VGG weights
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Store layers for feature extraction
        self.slices = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_layer:layer+1]))
            prev_layer = layer + 1
        
        self.slices = self.slices.to(device)
        
        # Normalization for VGG (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Move normalization buffers to device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
    
    def normalize(self, x):
        """Normalize from [-1, 1] to VGG input range."""
        # First convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Then normalize with ImageNet stats
        return (x - self.mean) / self.std
    
    def forward(self, fake, real):
        """
        Compute perceptual loss between fake and real images.
        
        Args:
            fake: Generated images (B, 3, H, W) in [-1, 1]
            real: Real images (B, 3, H, W) in [-1, 1]
        
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize inputs
        fake = self.normalize(fake)
        real = self.normalize(real)
        
        loss = 0
        
        # Extract features from each layer and compute L1 loss
        fake_feat = fake
        real_feat = real
        
        for slice_layer in self.slices:
            fake_feat = slice_layer(fake_feat)
            real_feat = slice_layer(real_feat)
            loss += torch.nn.functional.l1_loss(fake_feat, real_feat)
        
        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss using Discriminator intermediate features.
    
    This encourages the Generator to produce images that match
    the statistics of real images at multiple scales in D's feature space.
    
    Lighter than VGG - no extra model needed.
    """
    
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
    
    def forward(self, fake_features: list, real_features: list):
        """
        Compute feature matching loss.
        
        Args:
            fake_features: List of D features for fake images
            real_features: List of D features for real images
        
        Returns:
            Feature matching loss (scalar)
        """
        loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            loss += torch.nn.functional.l1_loss(fake_feat, real_feat.detach())
        return loss


if __name__ == "__main__":
    print("Testing Perceptual Loss...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test VGG Perceptual Loss
    perceptual_loss = VGGPerceptualLoss(device=device)
    
    # Create dummy images
    fake = torch.randn(4, 3, 128, 128, device=device)
    real = torch.randn(4, 3, 128, 128, device=device)
    
    loss = perceptual_loss(fake, real)
    print(f"Perceptual loss: {loss.item():.4f}")
    
    print("Perceptual loss test passed!")
