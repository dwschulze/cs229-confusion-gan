"""Generate virtual IHC images from H&E test patches using trained confusion-GAN.

Usage:
    python3 test_confusion_gan.py \
        --input_dir /home/ubuntu/cs229/data/Test_dataset/HnE/processed_data/256x256/ \
        --output_dir ./results/ \
        --checkpoint ./checkpoints/ConfusionGAN/latest_net_G_A.pth \
        --gpu_id 0
"""
import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.networks import define_G


def main():
    parser = argparse.ArgumentParser(description='Generate virtual IHC from H&E patches')
    parser.add_argument('--input_dir', required=True, help='Directory of H&E test patches (searched recursively)')
    parser.add_argument('--output_dir', default='./results', help='Output directory for generated IHC images')
    parser.add_argument('--checkpoint', required=True, help='Path to trained G_A checkpoint (.pth)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (default 256)')
    args = parser.parse_args()

    # Device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load generator
    G_A = define_G(3, 3, 64, 'unet_256', 'instance', True, 'normal', 0.02, [])
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    # Handle DataParallel wrapping (keys prefixed with 'module.')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    G_A.load_state_dict(state_dict)
    G_A.to(device)
    G_A.eval()
    print(f'Loaded generator from {args.checkpoint}')

    # Transforms (same as training: normalize to [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Find all input images
    exts = {'.png', '.jpg', '.jpeg'}
    input_paths = sorted([p for p in Path(args.input_dir).rglob('*') if p.suffix.lower() in exts])
    print(f'Found {len(input_paths)} input images')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate
    with torch.no_grad():
        for i, img_path in enumerate(input_paths):
            # Load and transform
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            # Generate virtual IHC
            fake_ihc = G_A(input_tensor)

            # Denormalize from [-1, 1] to [0, 1]
            fake_ihc = (fake_ihc.squeeze(0) * 0.5 + 0.5).clamp(0, 1)

            # Save
            output_name = img_path.stem + '_virtual_IHC.png'
            output_path = os.path.join(args.output_dir, output_name)
            transforms.ToPILImage()(fake_ihc.cpu()).save(output_path)

            if (i + 1) % 100 == 0:
                print(f'Processed {i + 1}/{len(input_paths)}')

    print(f'Done. {len(input_paths)} virtual IHC images saved to {args.output_dir}')


if __name__ == '__main__':
    from util.log_setup import setup_logging
    setup_logging('test_confusion_gan')
    main()
