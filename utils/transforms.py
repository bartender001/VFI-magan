import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image
import torch

class PairRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """Now handles single images instead of lists"""
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img
        
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return TF.crop(img, y1, x1, th, tw)

def get_train_transforms():
    return transforms.Compose([
        PairRandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])