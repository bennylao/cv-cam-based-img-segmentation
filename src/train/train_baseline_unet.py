import pathlib
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from skimage.morphology import binary_closing
from src import utils
from src.models.baseline_unet import UNet



if __name__ == "__main__":

    IMAGE_SIZE = 256

    # Configure paths
    ROOT = utils.get_root_dir(pathlib.Path(__file__).resolve())
    DATA_DIR = ROOT.joinpath("data")
    OUTPUT_DIR = ROOT.joinpath("output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load dataset
    train_transforms = utils.Compose([
        utils.PILToTensor(),
        utils.ResizeImgAndMask(size=(IMAGE_SIZE, IMAGE_SIZE)),
        utils.ConvertMaskToBinary(),
        utils.RandomHorizontalFlip(flip_prob=0.5),
        utils.RandomVerticalFlip(flip_prob=0.5),
        utils.RandomRotation(degrees=30),
        utils.ToDtype(dtype=torch.float32, scale=True),
    ])

    test_transforms = utils.Compose([
        utils.PILToTensor(),
        utils.ResizeImgAndMask(size=(IMAGE_SIZE, IMAGE_SIZE)),
        utils.ConvertMaskToBinary(),
        utils.ToDtype(dtype=torch.float32, scale=True),
    ])

    trainset, testset = utils.construct_dataset(
        data_dir=DATA_DIR,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    testloader = DataLoader(
        testset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
    )

    model = UNet(num_classes=2).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )



