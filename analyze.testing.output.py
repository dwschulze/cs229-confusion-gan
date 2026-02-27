"""Analyze confusion-GAN test output using the trained IHC classifier.

Runs the IHC classifier on generated virtual IHC images and reports
positive/negative counts, percentages, and confidence distributions.

Usage:
    python3 analyze.testing.output.py \
        --input_dir ./results/ \
        --pretrained_IHC_Classifier /path/to/ihc_classifier.pth \
        --gpu_id 0
"""
import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.IHC_Classifier import IHC_Classifier


def main():
    parser = argparse.ArgumentParser(description='Analyze generated IHC images with IHC classifier')
    parser.add_argument('--input_dir', required=True, help='Directory of generated virtual IHC images')
    parser.add_argument('--pretrained_IHC_Classifier', required=True, help='Path to trained IHC classifier checkpoint (.pth)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (default 256)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Threshold for pos/neg classification')
    args = parser.parse_args()

    # Device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load classifier
    classifier = IHC_Classifier(img_size=args.img_size).to(device)
    checkpoint = torch.load(args.pretrained_IHC_Classifier, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    print(f'Loaded IHC classifier from {args.pretrained_IHC_Classifier}')

    # Transforms (same normalization as training)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Find all images
    exts = {'.png', '.jpg', '.jpeg'}
    image_paths = sorted([p for p in Path(args.input_dir).rglob('*') if p.suffix.lower() in exts])
    print(f'Found {len(image_paths)} images to analyze')

    # Classify in batches
    positive_count = 0
    negative_count = 0
    pos_confidences = []
    neg_confidences = []
    high_confidence_pos = []  # top positive examples
    high_confidence_neg = []  # top negative examples

    with torch.no_grad():
        batch_imgs = []
        batch_paths = []

        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            batch_imgs.append(transform(img))
            batch_paths.append(img_path)

            if len(batch_imgs) == args.batch_size or i == len(image_paths) - 1:
                input_tensor = torch.stack(batch_imgs).to(device)
                output = classifier(input_tensor)  # [B, 2] — [P(pos), P(neg)]

                for j in range(len(batch_imgs)):
                    p_pos = output[j, 0].item()
                    p_neg = output[j, 1].item()

                    if p_pos > args.confidence_threshold:
                        positive_count += 1
                        pos_confidences.append(p_pos)
                        if p_pos > 0.9:
                            high_confidence_pos.append((p_pos, str(batch_paths[j])))
                    else:
                        negative_count += 1
                        neg_confidences.append(p_neg)
                        if p_neg > 0.9:
                            high_confidence_neg.append((p_neg, str(batch_paths[j])))

                batch_imgs = []
                batch_paths = []

                if (i + 1) % 1000 == 0:
                    print(f'  Processed {i + 1}/{len(image_paths)}...')

    total = positive_count + negative_count
    print(f'\n{"="*60}')
    print(f'RESULTS')
    print(f'{"="*60}')
    print(f'Total images analyzed: {total}')
    print(f'Positive (CK16+):     {positive_count} ({100*positive_count/total:.1f}%)')
    print(f'Negative (CK16-):     {negative_count} ({100*negative_count/total:.1f}%)')

    if pos_confidences:
        avg_pos = sum(pos_confidences) / len(pos_confidences)
        print(f'\nPositive confidence:  avg={avg_pos:.3f}  min={min(pos_confidences):.3f}  max={max(pos_confidences):.3f}')
    if neg_confidences:
        avg_neg = sum(neg_confidences) / len(neg_confidences)
        print(f'Negative confidence:  avg={avg_neg:.3f}  min={min(neg_confidences):.3f}  max={max(neg_confidences):.3f}')

    # Show some high-confidence examples
    high_confidence_pos.sort(reverse=True)
    high_confidence_neg.sort(reverse=True)

    if high_confidence_pos:
        print(f'\nTop 5 most confident POSITIVE images:')
        for conf, path in high_confidence_pos[:5]:
            print(f'  {conf:.4f}  {os.path.basename(path)}')

    if high_confidence_neg:
        print(f'\nTop 5 most confident NEGATIVE images:')
        for conf, path in high_confidence_neg[:5]:
            print(f'  {conf:.4f}  {os.path.basename(path)}')

    # Compare to expected ratio
    print(f'\n{"="*60}')
    print(f'COMPARISON TO REAL IHC')
    print(f'{"="*60}')
    print(f'Generated images positive rate: {100*positive_count/total:.1f}%')
    print(f'Real IHC positive rate (auto-labeled): ~43%')
    print(f'If these are close, the model has learned appropriate staining frequency.')


if __name__ == '__main__':
    from util.log_setup import setup_logging
    setup_logging('analyze.testing.output')
    main()
