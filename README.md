# Virtual Immunohistochemistry Staining for Histological Images Assisted by Weakly-supervised Learning
The official implementation of "Virtual Immunohistochemistry Staining for Histological Images Assisted by Weakly-supervised Learning".

## Setup for IntelARC xpu:
You'll have to use a python version <= 3.12 and manually install these packages.  First do a  

uv sync  

then install the following.  torch version 2.8 is needed because IPEX hasn't caught up to 2.10 yet.

uv:                                                                                                                                              
  uv pip install torch==2.8.0+xpu torchvision --index-url https://download.pytorch.org/whl/xpu --force-reinstall    
  uv pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --index-strategy unsafe-best-match                                                                                                                                  
                                                                                                                                                   
pip:                                                                                                                                             
  pip install torch==2.8.0+xpu torchvision --index-url https://download.pytorch.org/whl/xpu --force-reinstall                                        
  pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  

## How to use: 
```shell
python3 train.py --data_train_A ./dataset/trainA --data_train_B ./dataset/trainB \
    --load_size 256 --crop_size 256 --preprocess none --model confusion_gan --pretrained_IHC_Classifier ./pretrain_IHC_classifier.pth \
    --netG unet_256 --netD basic --netE basic_3d --A_labels ./trainA_labels.pt \
    --dataset_mode unaligned --direction AtoB 
```

🧱 **Built Upon**

This codebase is based on [ConfusionGAN](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Virtual_Immunohistochemistry_Staining_for_Histological_Images_Assisted_by_Weakly-supervised_Learning_CVPR_2024_paper.pdf) and its [source code](https://github.com/jiahanli2022/confusion-GAN)  
We also used the [FD-DINOv2](https://github.com/justin4ai/FD-DINOv2/tree/FD-DINOv2) metric.  
We thank the original authors for their contributions.

© This code is released under the GPLv3 license and is intended for non-commercial academic research only.
