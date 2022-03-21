# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : food101n.py
#   @Author      : Zeren Sun
#   @Created date: 2020/7/5 4:46 PM
#   @Description :
#
# ================================================================
import os
from torchvision.datasets import VisionDataset
from PIL import Image
from tqdm import tqdm


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def is_image_file(filename, extensions):
    return filename.lower().endswith(extensions)


def find_classes(root):
    root = os.path.expanduser(root)
    category_file = os.path.join(root, 'meta', 'classes.txt')
    classes = []
    with open(category_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('class_name'):
            continue
        classes.append(line.strip())
    classes.sort()
    assert len(classes) == 101, f'number of classes is expected to be 101, got {len(classes)}!'
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_datasets(root, class_to_idx, extensions):
    root = os.path.expanduser(root)
    instances = []
    labels = []
    with open(os.path.join(root, 'meta', 'imagelist.tsv'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        target_class, image_file_name = line.strip().split('/')
        if is_image_file(image_file_name, extensions):
            path = os.path.join(root, 'images', target_class, image_file_name)
            instances.append(path)
            labels.append(class_to_idx[target_class])
    return instances, labels


class Food101N(VisionDataset):
    # estimated noisy class label accuracy: 80%
    # image classification is evaluated on Food-101 test set
    # held-out classes: classes with no human supervision
    # samples: 310019
    # samples with verification label in train set: 52868
    # samples with verification label in valid set:  4741

    def __init__(self, root, use_cache=False, transform=None, target_transform=None, loader=pil_loader, extensions=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp') if extensions is None else extensions
        self.loader = loader
        self.use_cache = use_cache

        classes, class_to_idx = find_classes(root)
        samples, targets = make_datasets(root, class_to_idx, self.image_extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root))

        self.n_samples = len(samples)
        self.samples = samples
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.sample_to_index = {self.samples[i]: i for i in range(self.n_samples)}
        self.loaded_samples = self._cache_dataset() if self.use_cache else None

    def _cache_dataset(self):
        cached_samples = []
        print('caching samples ... ')
        for idx, path in enumerate(tqdm(self.samples, ncols=100, ascii=' >')):
            image = self.loader(path)
            cached_samples.append(image)
        assert len(cached_samples) == self.n_samples
        return cached_samples

    def __getitem__(self, index):
        if self.use_cache:
            assert len(self.loaded_samples) == self.n_samples
            sample, target = self.loaded_samples[index], self.targets[index]
        else:
            sample, target = self.loader(self.samples[index]), self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'index': index, 'data': sample, 'label': target}

    def __len__(self):
        return len(self.samples)

    def get_verified_samples(self, validation=False):
        verified_samples = {}
        root = os.path.expanduser(self.root)
        split = 'val' if validation else 'train'
        with open(os.path.join(root, 'meta', f'verified_{split}.tsv')) as f:
            lines = f.readlines()
        for line in lines:
            image_path, verified_label = line.strip().split('\t')  # verified label is 0 if label is incorrect, 1 otherwise
            image_path = os.path.join(root, 'images', image_path)
            if is_image_file(image_path, self.image_extensions):
                if image_path in self.sample_to_index.keys():
                    sample_index = self.sample_to_index[image_path]
                    verified_samples[sample_index] = verified_label
        return verified_samples


if __name__ == '__main__':
    train_data = Food101N('../Datasets/food101n')
    print(f'Train ---> {train_data.n_samples}')   # 310009
    # print(train_data.sample_to_index)
    verified_samples_train = train_data.get_verified_samples(validation=False)   # 52867, one line does not match,
                                                                                 # 'hot_and_sour_soup/a49e6fb26e356c6c41a37b5c18176355.jpg'
                                                                                 # is not present in imagelist.tsv
    verified_samples_valid = train_data.get_verified_samples(validation=True)    # 4741
    print(len(verified_samples_train), len(verified_samples_valid))
