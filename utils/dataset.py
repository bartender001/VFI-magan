import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class VimeoDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'sequences')
        self.transform = transform
        
        with open(os.path.join(root_dir, f'tri_{mode}list.txt'), 'r') as f:
            self.triplets = [
                os.path.join(self.root_dir, line.strip(), f'im{i}.png')
                for line in f for i in [1,2,3]
            ][::3]  # Group into triplets

    def __len__(self):
        return len(self.triplets) // 3

    def __getitem__(self, idx):
        prev, target, next = self.triplets[idx*3:(idx+1)*3]
        
        # Load frames individually
        prev_frame = Image.open(prev).convert('RGB')
        target_frame = Image.open(target).convert('RGB')
        next_frame = Image.open(next).convert('RGB')
        
        if self.transform:
            # Apply transform to each frame separately
            prev_frame = self.transform(prev_frame)
            target_frame = self.transform(target_frame)
            next_frame = self.transform(next_frame)
        
        input_frames = torch.cat([prev_frame, next_frame], dim=0)
        return input_frames, target_frame