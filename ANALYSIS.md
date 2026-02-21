# Confusion-GAN: Codebase Analysis

## Paper Reference

**Virtual Immunohistochemistry Staining for Histological Images Assisted by Weakly-supervised Learning** (CVPR 2024)

Li et al., Harbin Institute of Technology / Peking University Shenzhen Hospital

---

## Paper Summary

Confusion-GAN is an unsupervised method for virtual IHC staining (converting H&E histological images to IHC images) that achieves comparable performance to supervised methods without requiring paired training data. It introduces two key modules:

1. **Patch-level Pathology Information Extractor (PPIE)** — uses MIL top-k selection, a diffusion model (BBDM), and a dual-spherical anomaly detection loss to extract patch-level pos/neg labels from H&E images that only have WSI-level labels. This ensures pathological consistency during virtual staining.

2. **Multi-Branch Discriminator (MBD)** — combines a standard PatchGAN discriminator with a confusion discriminator that must identify a single generated image hidden among N-1 real images in a reference pool, forcing the generator to match the real distribution more closely.

A **Pathology Loss** (BCE between PPIE-derived H&E labels and IHC classifier predictions on generated images) constrains pathological consistency. The full generator loss is:

```
L_G = α·L_path + β·L_C_G + λ·L_Adv + η·L_Cycle + ι·L_Identity
```

With hyperparameters: α=1, β=1, λ=1, η=10, ι=5, pool size N=32, patch size 256×256.

Evaluated on three datasets (HCI/GPC3, BCI/HER2, MIST/ER), it outperforms unsupervised baselines and achieves results comparable to or better than supervised methods. Pathologists could only distinguish generated from real GPC3 images at 48-62% accuracy (near chance).

---

## Codebase Structure

```
confusion-GAN/
├── train.py                    # Main training loop
├── README.md
├── data/
│   ├── __init__.py
│   ├── base_dataset.py         # Abstract dataset + transforms
│   └── unaligned_dataset.py    # Unpaired H&E/IHC loader
├── models/
│   ├── __init__.py
│   ├── base_model.py           # Abstract model base class
│   ├── confusion_gan_model.py  # Core confusion-GAN implementation
│   ├── IHC_Classifier.py       # 6-conv pathology classifier (Cls_ihc)
│   ├── networks.py             # Generator/discriminator architectures
│   └── test_model.py           # Inference-only model
├── options/
│   ├── __init__.py
│   ├── base_options.py         # Shared CLI options
│   ├── train_options.py        # Training-specific options
│   └── test_options.py         # Test-specific options
├── util/
│   ├── __init__.py
│   ├── util.py                 # tensor2im, save_image helpers
│   ├── image_pool.py           # Historical image buffer for D stability
│   ├── html.py                 # HTML result gallery
│   ├── visualizer.py           # Visdom/wandb logging
│   └── get_data.py             # Dataset downloader
└── figures/
    ├── HE/                     # 14 sample H&E images (for README/demo)
    ├── IHC/                    # 14 sample IHC images
    └── Confusion-GAN/          # Generated result samples
```

---

## Paper → Code Mapping

| Paper Component | Code Location | Notes |
|---|---|---|
| **Generator G_A, G_B** (UNet256) | `networks.py:UnetGenerator` | 8 downsample blocks, skip connections, Tanh output |
| **Discriminator D_A, D_B** (PatchGAN) | `networks.py:NLayerDiscriminator` | Standard branch of MBD; 70×70 receptive field |
| **Confusion Discriminator C_A, C_B** | `networks.py:NLayerDiscriminator_3d` | Implemented as `netE_A`, `netE_B` |
| **IHC Classifier (Cls_ihc)** | `IHC_Classifier.py` | 6 conv layers + FCN, pretrained & frozen during GAN training |
| **PPIE (full pipeline)** | **Not in this repo** | Dual-spherical loss, BBDM diffusion, top-k MIL selection are separate pre-training steps; only the output labels are consumed via `--A_labels` |
| **Pathology Loss (Eq. 1)** | `confusion_gan_model.py` | `BCELoss(fake_B_label, A_label)` |
| **Confusion Loss (Eq. 7-10)** | `confusion_gan_model.py:insert_fea_loss()` | Core confusion mechanism |
| **Cycle Loss** | `confusion_gan_model.py` | L1 loss, weight η=10 |
| **Identity Loss** | `confusion_gan_model.py` | L1 loss, weight ι=5 |
| **GAN Adversarial Loss** | `networks.py:GANLoss` | LSGAN mode (MSE) |
| **Reference pool (N=32)** | `unaligned_dataset.py` | Returns `RS_IHC`/`RS_HE` (32 random patches per iteration) |

---

## Key File Details

### `train.py` — Main Training Entry Point

- Parses options via `TrainOptions`
- Creates dataset and model
- Main loop: iterates over epochs and batches
- Calls `model.optimize_parameters()` each iteration
- Saves checkpoints at specified frequencies
- Supports resume training via `--continue_train`

### `models/confusion_gan_model.py` — Core Implementation

This is the heart of the system. It defines six networks:

- `netG_A`: Generator H&E → IHC
- `netG_B`: Generator IHC → H&E
- `netD_A`: PatchGAN discriminator for IHC domain
- `netD_B`: PatchGAN discriminator for H&E domain
- `netE_A`: Confusion discriminator for H&E domain
- `netE_B`: Confusion discriminator for IHC domain
- `IHC_classifier`: Pretrained pathology classifier (frozen)

**Training alternates three optimization steps:**

1. Update generators G_A, G_B (with D and E frozen)
2. Update discriminators D_A, D_B (with G and E frozen)
3. Update confusion discriminators E_A, E_B (with D and G frozen)

**Loss components:**

| Loss | Variable | Function |
|---|---|---|
| GAN adversarial | `criterionGAN` | `GANLoss` (LSGAN / MSE) |
| Pathology | `Patho_loss` | `BCELoss` |
| Cycle consistency | `criterionCycle` | `L1Loss` |
| Identity | `criterionIdt` | `L1Loss` |

**The confusion mechanism (`insert_fea_loss()`):**

1. Extracts features from the generated image and 32 real reference images via the shared feature extractor (discriminator in `phase='triplesem'` mode)
2. Randomly inserts the fake's features among the real features
3. Concatenates along the channel dimension → `(1, N*C, H, W)`
4. The confusion discriminator must identify which entry is fake
5. The generator tries to make this identification impossible

### `models/networks.py` — Architecture Definitions

**Generators (`define_G`):**

- `unet_256` (default): UNet with 8 downsampling blocks for 256×256 input
- `unet_128`: UNet with 7 downsampling blocks
- `resnet_9blocks` / `resnet_6blocks`: ResNet-based alternatives

**UnetGenerator architecture:**

- Symmetric encoder-decoder with skip connections
- Downsampling: Conv2d + BatchNorm + LeakyReLU
- Upsampling: ConvTranspose2d + BatchNorm + ReLU + Dropout
- Output: Tanh activation ([-1, 1] range)

**Discriminators (`define_D`):**

- `basic` (default): 3-layer PatchGAN, 70×70 receptive field
- `n_layers`: Configurable depth PatchGAN
- `pixel`: 1×1 PixelGAN

**NLayerDiscriminator** has two forward modes:

- `phase='default'`: Full discriminator output (real/fake classification)
- `phase='triplesem'`: Feature extraction only (used by confusion discriminator)

**NLayerDiscriminator_3d** (confusion discriminator):

- Takes concatenated batch features: `(1, N*C, H, W)` input
- Single Conv2d layer for mixing/processing
- Output: `(1, N, h, w)` — per-patch authenticity predictions

### `models/IHC_Classifier.py` — Pathology Classifier

- 6 successive Conv2d layers (kernel=4, stride=2, padding=1)
- Channel progression: 3 → 16 → 32 → 64 → 128 → 256 → 512
- Linear layer → 2-class output (positive/negative)
- ReLU intermediate activations, Softmax output
- Pretrained on color-analysis-derived pos/neg IHC labels
- Frozen during confusion-GAN training

### `data/unaligned_dataset.py` — Dataset Loader

- Loads unpaired H&E and IHC 256×256 PNG patches
- Random pairing each iteration (no aligned pairs needed)
- Loads weak labels from `--A_labels` `.pt` file
- Returns 32 random reference samples (`RS_IHC`, `RS_HE`) per iteration for the confusion discriminator

Per-sample data returned:

```python
{
    'A': H&E image tensor,          # (3, 256, 256)
    'B': IHC image tensor,          # (3, 256, 256)
    'A_paths': str,
    'B_paths': str,
    'RS_IHC': [32 IHC tensors],     # Reference pool for confusion D
    'RS_HE': [32 H&E tensors],      # Reference pool for confusion D
    'A_label': tensor [1,0] or [0,1] # Weak label from PPIE
}
```

### `util/image_pool.py` — Historical Image Buffer

- Stores up to 50 previously generated images
- 50% chance to return a buffered image instead of the current one
- Standard CycleGAN technique for discriminator stability

---

## What's NOT in This Repo

The following are **separate pre-training steps** described in the paper but not included in this codebase:

1. **PPIE training pipeline** — the dual-spherical loss (Eq. 3-4), BBDM diffusion model for generating synthetic positive patches, top-k MIL selection, and the OOD/ID anomaly detection framework. Only the final output (pseudo-labels as a `.pt` file) is consumed by `--A_labels`.

2. **IHC Classifier pre-training** — trained separately on IHC patches with pos/neg labels derived from color analysis (>1% yellow-brown area = positive). Loaded as a frozen `.pth` checkpoint via `--pretrained_IHC_Classifier`.

3. **BBDM diffusion model** — used to generate synthetic positive H&E patches from negative ones during PPIE training.

---

## Data Requirements

To train confusion-GAN, you need:

| Requirement | CLI Flag | Description |
|---|---|---|
| H&E patches | `--data_train_A` | Directory of 256×256 `.png` H&E patches |
| IHC patches | `--data_train_B` | Directory of 256×256 `.png` IHC patches |
| Weak labels | `--A_labels` | `.pt` file mapping H&E filenames → `[1,0]`/`[0,1]` tensors |
| IHC classifier | `--pretrained_IHC_Classifier` | Pretrained `.pth` checkpoint |

The `figures/` directory contains 14 sample images each (H&E and IHC) — these are for the paper/README demo only, not for training.

### Available Datasets (from the paper)

| Dataset | Stain Conversion | Source | Scale |
|---|---|---|---|
| **HCI** (authors') | H&E → GPC3 | Hepatocellular carcinoma, Peking University Shenzhen Hospital | ~1.4M patches from 100 WSIs |
| **BCI** | H&E → HER2 | Breast cancer (Liu et al., CVPR 2022) | Public |
| **MIST** | H&E → ER | Breast cancer (Li et al., 2023) | Public |

---

## Training Command

```bash
python3 train.py \
  --data_train_A ./dataset/trainA \
  --data_train_B ./dataset/trainB \
  --load_size 256 --crop_size 256 \
  --preprocess none \
  --model confusion_gan \
  --pretrained_IHC_Classifier ./pretrain_IHC_classifier.pth \
  --netG unet_256 \
  --netD basic \
  --netE basic_3d \
  --A_labels ./trainA_labels.pt \
  --dataset_mode unaligned \
  --direction AtoB
```
