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
normal_std = (0.229, 0.224, 0.225)


def get_imagenet(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.RandomCrop(size=224,
                              padding=int(224*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    base_dataset = datasets.ImageNet(root, split="train")

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.labels)

    train_labeled_dataset = ImageNet_SSL(
        root, train_labeled_idxs, split='train',
        transform=transform_labeled)

    train_unlabeled_dataset = ImageNet_SSL(
        root, train_unlabeled_idxs, split='train',
        transform=TransformFixMatchImageNet(mean=normal_mean, std=normal_std))

    test_dataset = datasets.ImageNet(
        root, split="val", transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class ImageNet_SSL(datasets.ImageNet):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.targets = self.labels
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TransformFixMatchImageNet(TransformFixMatch):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

ImageNet_GETTERS = {'imagenet': get_imagenet}

