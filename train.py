#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from models import SuperGenerator, PatchDiscriminator
from models.losses import PerceptualLoss
from utils.dataset import VimeoDataset
from utils.transforms import get_train_transforms, get_val_transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import argparse
import time
import os

# MacOS-specific settings
mp.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/vimeo_triplet', help='Dataset root path')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--save_dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=2, 
                      help='Number of data loading workers (recommend 2 for MacOS)')
    return parser.parse_args()

def setup_environment(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def create_dataloaders(args):
    train_set = VimeoDataset('data/vimeo_triplet', 'train', get_train_transforms())
    val_set = VimeoDataset('data/vimeo_triplet', 'test', get_val_transforms())
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Start with 0 workers for debugging
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=min(2, args.num_workers),  # Fewer workers for validation
        pin_memory=True
    )
    
    return train_loader, val_loader

def initialize_models(device):
    G = SuperGenerator().to(device)
    D = PatchDiscriminator().to(device)
    
    # Apply weight initialization
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
    G.apply(weights_init)
    D.apply(weights_init)
    
    return G, D

def train_epoch(G, D, train_loader, device, optimizer_G, optimizer_D, scaler, perceptual_loss, epoch, args):
    G.train()
    D.train()
    
    for i, (input_frames, real_frames) in enumerate(train_loader):
        input_frames = input_frames.to(device)
        real_frames = real_frames.to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        with torch.no_grad():
            fake_frames = G(input_frames)
        
        real_pred = D(real_frames)
        fake_pred = D(fake_frames.detach())
        
        loss_D_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)*0.9)  # Label smoothing
        loss_D_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        # Instead of mixed-precision, let's keep it simple for MPS device
        fake_frames = G(input_frames)
        fake_pred = D(fake_frames)
        
        loss_G_gan = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        loss_G_l1 = F.l1_loss(fake_frames, real_frames) * 100
        loss_G_perceptual = perceptual_loss(fake_frames, real_frames) * 0.1
        
        loss_G = loss_G_gan + loss_G_l1 + loss_G_perceptual
        
        # Skip gradient scaling for simplicity
        loss_G.backward()
        optimizer_G.step()
        
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(train_loader)}] "
                  f"D_loss: {loss_D.item():.4f} G_loss: {loss_G.item():.4f}")

def main():
    args = parse_args()
    device = setup_environment(args)
    train_loader, val_loader = create_dataloaders(args)
    G, D = initialize_models(device)
    
    # Create and move the perceptual loss to the device
    perceptual_loss = PerceptualLoss().to(device)
    
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr*2, betas=(0.5, 0.999))
    
    # For simplicity, let's avoid using GradScaler with MPS
    scaler = None

    for epoch in range(args.epochs):
        start_time = time.time()
        train_epoch(G, D, train_loader, device, optimizer_G, optimizer_D, scaler, perceptual_loss, epoch, args)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print(f"Epoch {epoch} completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == '__main__':
    main()