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

## Reference
If you found our work useful in your research, please consider citing our works(s) at:
```
@inproceedings{li2024virtual,
  title={Virtual immunohistochemistry staining for histological images assisted by weakly-supervised learning},
  author={Li, Jiahan and Dong, Jiuyang and Huang, Shenjin and Li, Xi and Jiang, Junjun and Fan, Xiaopeng and Zhang, Yongbing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11259--11268},
  year={2024}
}
```

🧱 **Built Upon**

Parts of this codebase are adapted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).  
We thank the original authors for their contributions.

© This code is released under the GPLv3 license and is intended for non-commercial academic research only.
