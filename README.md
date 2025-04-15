# Frame Interpolation with SuperGenerator

A deep learning-based video frame interpolation system that uses GANs to generate high-quality intermediate frames from consecutive video frames. This project can be used to increase video frame rates, create slow-motion effects, or enhance video smoothness.

## Features

- High-quality frame interpolation using a custom SuperGenerator architecture
- GAN-based training for realistic frame generation
- Perceptual loss for preserving visual details
- Support for Apple Silicon (MPS) and CPU devices
- Easy-to-use inference script for interpolating between any two frames

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/frame-interpolation-gan.git
cd frame-interpolation-gan
```

### Requirements

- Python 3.7+
- PyTorch 2.0+
- Pillow
- torchvision

Install the required packages:

```bash
pip install torch torchvision pillow
```

## Dataset

This model was trained on the [Vimeo-90K Triplet dataset](http://toflow.csail.mit.edu/). To train with this dataset:

1. Download the dataset from [http://toflow.csail.mit.edu/](http://toflow.csail.mit.edu/)
2. Extract the files to the `data/vimeo_triplet` directory
3. The dataset should have the following structure:

```
data/vimeo_triplet/
├── sequences/
│   ├── 00001/
│   │   ├── 0001/
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   └── im3.png
│   │   └── ...
│   └── ...
├── tri_trainlist.txt
└── tri_testlist.txt
```

## Model Architecture

The SuperGenerator model uses an encoder-decoder architecture with residual blocks and skip connections:

- 6-channel input (concatenated frames)
- Encoder with downsampling and residual blocks
- Bottleneck with residual connections
- Decoder with upsampling and skip connections from encoder
- PatchDiscriminator for adversarial training

## Training

To train the model:

```bash
python train.py --data_root data/vimeo_triplet --batch_size 8 --epochs 100
```

Additional training options:
- `--lr`: Learning rate (default: 2e-4)
- `--save_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--num_workers`: Number of data loading workers (default: 2)

## Inference

To interpolate between two frames:

```bash
python interpolate.py --prev path/to/frame1.png --next path/to/frame2.png --output output.png
```

Options:
- `--model`: Path to the trained model (default: 'generator.pth')
- `--output`: Path to save the interpolated frame (default: 'output.png')


## Acknowledgments
- Thanks to the creators of the Vimeo-90K dataset
