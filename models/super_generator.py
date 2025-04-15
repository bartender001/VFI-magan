# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.InstanceNorm2d(channels),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.InstanceNorm2d(channels)
#         )

#     def forward(self, x):
#         return x + self.block(x)

# class SuperGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(6, 64, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(64)
#         )
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(128)
#         )
        
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             ResidualBlock(128),
#             ResidualBlock(128),
#             nn.Conv2d(128, 256, 3, padding=1)
#         )
        
#         # Decoder
#         self.dec1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(),
#             ResidualBlock(128)
#         )
#         self.dec2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(),
#             ResidualBlock(64)
#         )
        
#         # Add channel adjustment for skip connection
#         self.skip_conv = nn.Conv2d(128, 64, 1)  # 1x1 conv to match channels
        
#         self.final = nn.Sequential(
#             nn.Conv2d(64, 3, 3, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         e1 = self.enc1(x)  # [B, 64, H/2, W/2]
#         e2 = self.enc2(e1) # [B, 128, H/4, W/4]
        
#         b = self.bottleneck(e2)  # [B, 256, H/4, W/4]
        
#         d1 = self.dec1(b)  # [B, 128, H/2, W/2]
        
#         # Adjust channels before skip connection
#         d1_adjusted = self.skip_conv(d1)  # [B, 64, H/2, W/2]
#         d2 = self.dec2(d1_adjusted + e1)  # Now both are 64 channels
        
#         return self.final(d2)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class SuperGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, padding=1)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128)
        )
        
        # Skip connection adjustment
        self.skip_conv = nn.Conv2d(64, 128, 1)  # Adjust encoder1 features to match decoder2
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, 64, 128, 128]
        e2 = self.enc2(e1) # [B, 128, 64, 64]
        
        # Bottleneck
        b = self.bottleneck(e2)  # [B, 256, 64, 64]
        
        # Decoder
        d1 = self.dec1(b)  # [B, 128, 128, 128]
        
        # Adjust skip connection and add
        e1_adjusted = self.skip_conv(e1)  # [B, 128, 128, 128]
        d2_input = d1 + e1_adjusted  # Both now [B, 128, 128, 128]
        
        d2 = self.dec2(d2_input)  # [B, 64, 256, 256]
        
        return self.final(d2)