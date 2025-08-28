# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# from models import SuperGenerator # Assuming 'models' directory is accessible
# import argparse

# def interpolate(prev_path, next_path, model_path):
#     """
#     Interpolates a frame between two input frames using a trained generator model.

#     Args:
#         prev_path (str): File path to the previous frame.
#         next_path (str): File path to the next frame.
#         model_path (str): File path to the saved model checkpoint.
#     """
#     # Set the device (mps for Apple Silicon, cuda for NVIDIA, or cpu)
#     if torch.backends.mps.is_available():
#         device = 'mps'
#     elif torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
    
#     print(f"Using device: {device}")

#     # Define the image transformation. 
#     # The training script uses ToTensor which scales images to [0, 1].
#     transform = transforms.ToTensor()
    
#     # --- Load and transform input frames ---
#     # Open images, convert to RGB, apply transform, and add a batch dimension
#     prev_img = transform(Image.open(prev_path).convert('RGB')).unsqueeze(0)
#     next_img = transform(Image.open(next_path).convert('RGB')).unsqueeze(0)
    
#     # Concatenate the two frames along the channel dimension (dim=1)
#     # The model expects a 6-channel input (3 for prev_img, 3 for next_img)
#     inputs = torch.cat([prev_img, next_img], dim=1).to(device)
    
#     # --- Load the trained model ---
#     G = SuperGenerator().to(device)
    
#     # The training script saves a dictionary, so we need to load the
#     # state dictionary from the 'G_state_dict' key.
#     checkpoint = torch.load(model_path, map_location=device)
#     G.load_state_dict(checkpoint['G_state_dict'])
    
#     # Set the model to evaluation mode
#     G.eval()
    
#     # --- Generate the intermediate frame ---
#     with torch.no_grad(): # Disable gradient calculation for inference
#         output = G(inputs)
    
#     # --- Convert the output tensor back to a PIL Image ---
#     # The model outputs a tensor in the [0, 1] range.
#     # 1. Squeeze the batch dimension.
#     # 2. Permute from (C, H, W) to (H, W, C) for numpy.
#     # 3. Move tensor to CPU.
#     # 4. Scale from [0, 1] to [0, 255].
#     # 5. Convert to an unsigned 8-bit integer array.
#     output_np = (output.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    
#     return Image.fromarray(output_np)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Frame Interpolation using a trained VFI-GAN model.")
#     parser.add_argument('--prev', type=str, required=True, help="Path to the first input frame (e.g., 'im1.png').")
#     parser.add_argument('--next', type=str, required=True, help="Path to the second input frame (e.g., 'im3.png').")
#     parser.add_argument('--model', type=str, default='checkpoints/checkpoint_epoch_100.pth', help="Path to the generator model checkpoint.")
#     parser.add_argument('--output', type=str, default='output.png', help="Path to save the interpolated output image.")
#     args = parser.parse_args()
    
#     # Generate the interpolated frame and save it
#     result_image = interpolate(args.prev, args.next, args.model)
#     result_image.save(args.output)
#     print(f"Interpolated frame saved to {args.output}")

import torch
from PIL import Image
import torchvision.transforms as transforms
from models import SuperGenerator # Assuming 'models' directory is accessible
import argparse

# Define the expected image size, which should match the training CROP_SIZE.
# This is the key change to make the script more robust.
EXPECTED_IMG_SIZE = (256, 256) 

def interpolate(prev_path, next_path, model_path):
    """
    Interpolates a frame between two input frames using a trained generator model.

    Args:
        prev_path (str): File path to the previous frame.
        next_path (str): File path to the next frame.
        model_path (str): File path to the saved model checkpoint.
    """
    # Set the device (mps for Apple Silicon, cuda for NVIDIA, or cpu)
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")

    # Define the image transformation pipeline.
    # 1. Resize images to the size the model was trained on.
    # 2. Convert images to PyTorch tensors, scaling pixels to the [0, 1] range.
    transform = transforms.Compose([
        transforms.Resize(EXPECTED_IMG_SIZE),
        transforms.ToTensor()
    ])
    
    # --- Load and transform input frames ---
    # Open images, convert to RGB, apply transform, and add a batch dimension
    prev_img = transform(Image.open(prev_path).convert('RGB')).unsqueeze(0)
    next_img = transform(Image.open(next_path).convert('RGB')).unsqueeze(0)
    
    # Concatenate the two frames along the channel dimension (dim=1)
    # The model expects a 6-channel input (3 for prev_img, 3 for next_img)
    inputs = torch.cat([prev_img, next_img], dim=1).to(device)
    
    # --- Load the trained model ---
    G = SuperGenerator().to(device)
    
    # The training script saves a dictionary, so we load the
    # state dictionary from the 'G_state_dict' key.
    checkpoint = torch.load(model_path, map_location=device)
    G.load_state_dict(checkpoint['G_state_dict'])
    
    # Set the model to evaluation mode (important for layers like BatchNorm)
    G.eval()
    
    # --- Generate the intermediate frame ---
    with torch.no_grad(): # Disable gradient calculation for inference
        output = G(inputs)
    
    # --- Convert the output tensor back to a PIL Image ---
    # 1. Squeeze the batch dimension.
    # 2. Permute from (C, H, W) to (H, W, C) for numpy.
    # 3. Move tensor to CPU.
    # 4. Scale from [0, 1] to [0, 255].
    # 5. Convert to an unsigned 8-bit integer array.
    output_np = (output.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    
    return Image.fromarray(output_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Frame Interpolation using a trained VFI-GAN model.")
    parser.add_argument('--prev', type=str, required=True, help="Path to the first input frame (e.g., 'im1.png').")
    parser.add_argument('--next', type=str, required=True, help="Path to the second input frame (e.g., 'im3.png').")
    parser.add_argument('--model', type=str, default='checkpoints/checkpoint_epoch_100.pth', help="Path to the generator model checkpoint.")
    parser.add_argument('--output', type=str, default='output.png', help="Path to save the interpolated output image.")
    args = parser.parse_args()
    
    # Generate the interpolated frame and save it
    result_image = interpolate(args.prev, args.next, args.model)
    result_image.save(args.output)
    print(f"Interpolated frame saved to {args.output}")