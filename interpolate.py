import torch
from PIL import Image
from models import SuperGenerator
from utils.transforms import get_transforms

def interpolate(prev_path, next_path, model_path):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    transform = get_transforms('val')
    
    # Load frames
    prev = transform(Image.open(prev_path).convert('RGB')).unsqueeze(0)
    next = transform(Image.open(next_path).convert('RGB')).unsqueeze(0)
    inputs = torch.cat([prev, next], dim=1).to(device)
    
    # Load model
    G = SuperGenerator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    
    # Generate
    with torch.no_grad():
        output = G(inputs)
    
    # Convert to image
    output = (output.squeeze().permute(1,2,0).cpu().numpy() * 127.5 + 127.5).astype('uint8')
    return Image.fromarray(output)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prev', required=True)
    parser.add_argument('--next', required=True)
    parser.add_argument('--model', default='generator.pth')
    parser.add_argument('--output', default='output.png')
    args = parser.parse_args()
    
    result = interpolate(args.prev, args.next, args.model)
    result.save(args.output)