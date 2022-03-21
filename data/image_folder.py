# -*- coding: utf-8 -*-

from torchvision.datasets import DatasetFolder
from PIL import Image
from tqdm import tqdm


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class IndexedImageFolder(DatasetFolder):
    def __init__(self, root, use_cache=False, transform=None, target_transform=None,
                 loader=pil_loader, is_valid_file=None):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        self.imgs = self.samples  # list, element is (path, label)

        self.use_cache = use_cache
        if self.use_cache:
            self.loaded_samples = self._cache_dataset()  # list, element is (PIL image, label)
        else:
            self.loaded_samples = None

    def _cache_dataset(self):
        cached_dataset = []
        n_samples = len(self.samples)
        print('caching samples ... ')
        for idx, sample in enumerate(tqdm(self.samples, ncols=100, ascii=' >')):
            path, target = sample
            image = self.loader(path)
            item = (image, target)
            cached_dataset.append(item)
        assert len(cached_dataset) == n_samples
        return cached_dataset

    def __getitem__(self, index):
        if self.use_cache:
            assert len(self.loaded_samples) == len(self.samples)
            sample, target = self.loaded_samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'index': index, 'data': sample, 'label': target}

