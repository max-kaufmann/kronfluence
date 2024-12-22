import copy
import math
from typing import List, Optional

import datasets
import numpy as np
import torch
import torchvision
from torch import nn


class Mul(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


def construct_mnist_classifier() -> nn.Module:
    # ResNet-9 architecture from https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb.
    def conv_bn(
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(),
        )

    model = torch.nn.Sequential(
        conv_bn(1, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        conv_bn(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Flatten(),
        torch.nn.Linear(64 * 7 * 7, 10),  # We MaxPooled a 28x28 image twice, so the output is 7x7.
    )
    return model


MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_mnist_dataset(
    split: str,
    class_with_box: int = 0,
    dataset_dir: str = "data/",
) -> datasets.Dataset:
    """Construct the MNIST dataset, but make some of the images have a distrinctive white box in the bottom right."""
    assert split in ["train", "eval_train", "valid"]

    normalize_transform = torchvision.transforms.Normalize(mean=(MNIST_MEAN,), std=(MNIST_STD,))
    transforms = [torchvision.transforms.ToTensor(), normalize_transform]

    if split == "train":
        transforms = [torchvision.transforms.RandomHorizontalFlip()] + transforms
    
    dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        download=True,
        train=split in ["train", "eval_train"],
        transform=torchvision.transforms.Compose(transforms),
    )

    return dataset
