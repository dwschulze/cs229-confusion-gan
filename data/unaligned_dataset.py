import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import torch
from pathlib import Path

def make_dataset(dir_path):
    # return sorted([str(p) for p in Path(dir_path).rglob('*.png')])
    extensions = {'.png', '.jpg', '.jpeg'}
    return sorted([str(p) for p in Path(dir_path).rglob('*') if p.suffix.lower() in extensions])

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        assert opt.phase == 'train'

        self.dir_A = opt.data_train_A
        self.dir_B = opt.data_train_B
        print('[TRACE] Scanning dataset A paths...', flush=True)
        self.A_paths = make_dataset(self.dir_A)
        print('[TRACE] Found %d A paths. Scanning dataset B paths...' % len(self.A_paths), flush=True)
        self.B_paths = make_dataset(self.dir_B)
        print('[TRACE] Found %d B paths.' % len(self.B_paths), flush=True)

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        print('[TRACE] Loading A_labels...', flush=True)
        self.A_labels = torch.load(opt.A_labels) # opt.A_labels is a dictionary
        print('[TRACE] A_labels loaded. Creating transforms...', flush=True)
        self.transform_A = get_transform(self.opt, grayscale=False)
        self.transform_B = get_transform(self.opt, grayscale=False)
        self.transform = get_transform(self.opt, grayscale=False)

        # Pre-load and pre-transform all images into RAM to avoid
        # repeated disk I/O for the 32 reference images per iteration
        print('Pre-loading dataset A (%d images)...' % self.A_size, flush=True)
        self.A_cache = []
        for i, p in enumerate(self.A_paths):
            if i == 0:
                print('[TRACE] Loading first image: %s' % p, flush=True)
                img = Image.open(p).convert('RGB')
                print('[TRACE] Image opened, applying transform...', flush=True)
                t = self.transform_A(img)
                print('[TRACE] Transform done, shape=%s' % str(t.shape), flush=True)
                self.A_cache.append(t)
            else:
                if i <= 5:
                    print('[TRACE] Loading image %d: %s' % (i, p), flush=True)
                self.A_cache.append(self.transform_A(Image.open(p).convert('RGB')))
                if i <= 5:
                    print('[TRACE] Image %d done' % i, flush=True)
            if (i + 1) % 1000 == 0:
                print('  A: %d / %d' % (i + 1, self.A_size), flush=True)
        print('Pre-loading dataset B (%d images)...' % self.B_size, flush=True)
        self.B_cache = []
        for i, p in enumerate(self.B_paths):
            self.B_cache.append(self.transform_B(Image.open(p).convert('RGB')))
            if (i + 1) % 1000 == 0:
                print('  B: %d / %d' % (i + 1, self.B_size), flush=True)
        print('Pre-loading complete.', flush=True)

    def __getitem__(self, index):
        A_path = self.A_paths[index]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')

        # assert A_img.size[0] == 256 and A_img.size[0] == 256 and B_img.size[0] == 256 and B_img.size[0] == 256

        A = self.A_cache[index]
        B = self.B_cache[index_B]

        Rs_IHC, Rs_HE = [], []
        for _ in range(32):
            index_RA = random.randint(0, self.A_size - 1)
            index_RB = random.randint(0, self.B_size - 1)

            # Rs_HE.append(self.transform(Image.open(self.A_paths[index_RA]).convert('RGB')))
            # Rs_IHC.append(self.transform(Image.open(self.B_paths[index_RB]).convert('RGB')))
            Rs_HE.append(self.A_cache[index_RA])
            Rs_IHC.append(self.B_cache[index_RB])

        if int(self.A_labels[A_path]) ==1: # label adapt from One-Class-Classifier
            A_label = torch.Tensor([1, 0])  # OCC positive or Custom
        else:
            A_label = torch.Tensor([0, 1])  # OCC negative or Custom

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'RS_IHC':Rs_IHC, 'RS_HE':Rs_HE, 'A_label':A_label}

    def __len__(self):
        return min(self.A_size, self.B_size)



