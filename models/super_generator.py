import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.ss2d_block.py import SS2D, ChannelAttention, MambaBlock, ResidualBlock, VSSBlock



####
####
####
#### modified at more than just the bottleneck
class SuperGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(64) # Keep convs in early layers for local features
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            MambaBlock(128) # <-- MODIFIED: Use MambaBlock instead of ResidualBlock
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            MambaBlock(128), # <-- MODIFIED: Can be a MambaBlock too
            ChannelAttention(128),
            nn.Conv2d(128, 256, 3, padding=1)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            MambaBlock(128) # <-- MODIFIED: Use MambaBlock instead of ResidualBlock
        )
        
        # Skip connection adjustment
        self.skip_conv = nn.Conv2d(64, 128, 1)
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64) # Keep convs in final layer for detail refinement
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        """
        Defines the forward pass of the SuperGenerator model.

        Args:
            x (torch.Tensor): The input tensor, which is a concatenation of
                              the previous and next frames. Shape: (B, 6, H, W).

        Returns:
            torch.Tensor: The generated intermediate frame. Shape: (B, 3, H, W).
        """
        # --- Encoder Path ---
        # e1 will be used for the skip connection
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        # --- Bottleneck ---
        b = self.bottleneck(e2)

        # --- Decoder Path with Skip Connection ---
        # First decoder block
        d1 = self.dec1(b)

        # Adjust the channels of the first encoder's output to match the decoder's
        skip_connection = self.skip_conv(e1)

        # Add the skip connection to the decoder's output
        d1_with_skip = d1 + skip_connection

        # Second decoder block
        d2 = self.dec2(d1_with_skip)

        # --- Final Output Layer ---
        # Generate the final 3-channel image
        output = self.final(d2)

        return output




