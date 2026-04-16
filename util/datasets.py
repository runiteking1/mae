# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class HuggingFaceImageNet(Dataset):
    """Wraps a HuggingFace datasets.Dataset (loaded from parquet) as a PyTorch Dataset."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        label = row["label"]
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.open(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_dataset_hf(is_train, args):
    """Build dataset from HuggingFace parquet files at args.data_path."""
    from datasets import load_dataset

    split = "train" if is_train else "validation"
    hf_ds = load_dataset(args.data_path, split=split)
    transform = build_transform(is_train, args)
    dataset = HuggingFaceImageNet(hf_ds, transform=transform)
    print(f"HuggingFace {split} dataset: {len(dataset)} samples")
    return dataset


def build_dataset_hf_pretrain(is_train, args):
    """Build pre-training dataset (simple augmentation) from HuggingFace parquet files."""
    from datasets import load_dataset

    split = "train" if is_train else "validation"
    hf_ds = load_dataset(args.data_path, split=split)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = HuggingFaceImageNet(hf_ds, transform=transform)
    print(f"HuggingFace {split} dataset: {len(dataset)} samples")
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
