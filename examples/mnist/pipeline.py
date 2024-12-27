import copy
import math
from typing import List, Optional, Literal

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


class InMemoryMNIST(torchvision.datasets.MNIST):
    """To avoid consistently moving tensors to the GPU, and given the small size of MNIST, we create a version of MNIST which is held in memory (GPU or Local)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data.to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Move the data to the GPU if avaliable
        self.data = self.data.unsqueeze(1)  # Add a channel dimension

    def __getitem__(self, index: int):
        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            target = self.target_transform(self.targets[index])

        return img, target


def add_box_to_mnist_dataset(
    dataset: datasets.Dataset,
    class_with_box: int = 0,
    box_size: int = 7,
) -> datasets.Dataset:
    """Add a white box to the bottom right of the images in the dataset."""
    class_indices = np.where(np.array(dataset.targets) == class_with_box)[0]
    dataset.data[class_indices, -box_size:, -box_size:] = 1.0
    return dataset


def get_mnist_dataset(
    split: Literal["train", "eval_train", "test"],
    class_with_box: int | None = 0,
    box_size: int = 7,
    dataset_dir: str = "data/",
    in_memory: bool = True,
) -> datasets.Dataset:
    """Construct the MNIST dataset, but make some of the images have a distrinctive white box in the bottom right."""
    assert split in ["train", "eval_train", "test"]

    transforms = [torchvision.transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]

    if in_memory:
        dataset = InMemoryMNIST(
            root=dataset_dir,
            download=True,
            train=split in ["train", "eval_train"],
            transforms=torchvision.transforms.Compose(transforms),
        )
    else:
        transforms = [torchvision.transforms.ToTensor()] + transforms
        dataset = torchvision.datasets.MNIST(
            root=dataset_dir,
            download=True,
            train=split in ["train", "eval_train"],
            transforms=torchvision.transforms.Compose(transforms),
        )

    # For the selected class, add a white box to the bottom right of the image.
    if class_with_box is not None:
        dataset = add_box_to_mnist_dataset(dataset, class_with_box, box_size)

    return dataset
