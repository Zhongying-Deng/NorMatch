import os
import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
from .cifar import x_u_split, TransformFixMatch
logger = logging.getLogger(__name__)

normal_mean = (0.485, 0.456, 0.406)
normal_std = (0.229, 0.84, 0.225)


def get_mini_imagenet(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(84),
        transforms.RandomCrop(size=84,
                              padding=int(84*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    base_dataset = datasets.ImageFolder(os.path.join(root, "train"))

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = MiniImageNet_SSL(
        root, train_labeled_idxs, split='train',
        transform=transform_labeled)

    train_unlabeled_dataset = MiniImageNet_SSL(
        root, train_unlabeled_idxs, split='train',
        transform=TransformFixMatchImageNet(mean=normal_mean, std=normal_std))

    test_dataset = datasets.ImageFolder(
        os.path.join(root, "test"), transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class MiniImageNet_SSL(datasets.ImageFolder):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(os.path.join(root, split), transform=transform,
                         target_transform=target_transform,
                         )
        if indexs is not None:
            self.data = []
            for i in indexs:
                path = self.samples[i][0]
                img = Image.open(path)
                img.convert('RGB')
                self.data.append(img)
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TransformFixMatchImageNet(TransformFixMatch):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(84),
            transforms.RandomCrop(size=84,
                                  padding=int(84*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(84),
            transforms.RandomCrop(size=84,
                                  padding=int(84*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

MiniImageNet_GETTERS = {'mini_imagenet': get_mini_imagenet}

