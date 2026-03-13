"""

Generate virtual IHC images from H&E test images using trained model weights from the checkpoints/ directory
This only uses the G_A generator.

Usage:
    python test_confusion_gan.py \
      --output_dir ./test.results.dean/test_confusion_gan/confusion-gan-256-fp16-lr_D-lambda_idt-1/latest \
      --checkpoint ./checkpoints/confusion-gan-256-fp16-lr_D-lambda_idt-1/latest_net_G_A.pth

ToDo:
    Need to save a mapping from the source HnE image file name to the generated fake IHC file name.

    Add support for batch size

"""
import argparse
import os
from datetime import datetime
import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.networks import define_G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='../../../data/cs229_final_project/Test_dataset/HnE/processed_data/256x256/')
    parser.add_argument('--output_dir', default='./test.results/test_confusion_gan/confusion-gan-256-fp16-lambda_idt-0')
    parser.add_argument('--checkpoint', default='./checkpoints/confusion-gan-256-fp16-lambda_idt-0/latest_net_G_A.pth')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--netG', type=str, default='unet_256')
    parser.add_argument('--no_dropout', action='store_true')
    args = parser.parse_args()

    print('Parameters:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    args.output_dir = os.path.join(args.output_dir, timestamp)

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    use_dropout = not args.no_dropout
    G_A = define_G(3, 3, 64, args.netG, 'instance', use_dropout, 'normal', 0.02, [])
    state_dict = torch.load(args.checkpoint, map_location='cpu')

    # strip leading "module"
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    G_A.load_state_dict(state_dict)
    G_A.to(device)
    G_A.eval()
    print(f'Loaded generator from {args.checkpoint}')

    #  normalize to [-1, 1].  this is for consistency with tanh
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    exts = {'.png', '.jpg', '.jpeg'}
    input_paths = sorted([p for p in Path(args.input_dir).rglob('*') if p.suffix.lower() in exts])
    print(f'Found {len(input_paths)} input images')

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, img_path in enumerate(input_paths):
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            fake_ihc = G_A(input_tensor)

            fake_ihc = (fake_ihc.squeeze(0) * 0.5 + 0.5).clamp(0, 1)

            output_name = img_path.stem + '_virtual_IHC.png'
            output_path = os.path.join(args.output_dir, output_name)
            transforms.ToPILImage()(fake_ihc.cpu()).save(output_path)

            if (i + 1) % 100 == 0:
                print(f'Processed {i + 1}/{len(input_paths)}')

    print(f'done. saved {len(input_paths)} virtual IHC images to {args.output_dir}')


if __name__ == '__main__':
    from util.log_setup import setup_logging
    setup_logging('test_confusion_gan', subdir='test_confusion_gan')
    main()
