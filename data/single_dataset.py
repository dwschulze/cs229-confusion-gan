from data.base_dataset import BaseDataset, get_transform
from pathlib import Path
from PIL import Image


def make_dataset(dir_path):
    extensions = {'.png', '.jpg', '.jpeg'}
    return sorted([str(p) for p in Path(dir_path).rglob('*') if p.suffix.lower() in extensions])


class SingleDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.A_paths = make_dataset(opt.dataroot)
        self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)
