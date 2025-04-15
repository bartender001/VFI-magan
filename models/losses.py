import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the updated weights parameter instead of pretrained
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
    
    def forward(self, fake, real):
        # Check if any parameter in the VGG model is on a different device
        # than the input tensors, and move the module if needed
        if next(self.vgg.parameters()).device != fake.device:
            self.vgg = self.vgg.to(fake.device)
            
        fake = (fake + 1) / 2  # [-1,1] -> [0,1]
        real = (real + 1) / 2
        
        # Handle grayscale if needed
        if fake.shape[1] == 1:
            fake = torch.cat([fake]*3, dim=1)
            real = torch.cat([real]*3, dim=1)
            
        return F.l1_loss(self.vgg(fake), self.vgg(real))

class FlowConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, fake, prev, next):
        # Placeholder - implement with RAFT if needed
        return torch.tensor(0.0, device=fake.device)