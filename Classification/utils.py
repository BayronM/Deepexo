#!/usr/bin/env python3

import torch
import torchvision
import numpy as np
from dataset import ExoDataset, data_preprocess
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_images,
    psf_images,
    c_ratio,
    test_images,
    batch_size,
    train_transform,
    num_workers=4,
    pin_memory=True
    ):

    preprocessed_train, train_label , mask_train = data_preprocess(np.load(train_images),
                                                                   np.load(psf_images),
                                                                   c_ratio=c_ratio,
                                                                   no_blend = True)
    preprocessed_test, test_label, mask_test =  data_preprocess(np.load(test_images),
                                                                np.load(psf_images),
                                                                c_ratio=c_ratio,
                                                                no_blend = True)
    trainset = ExoDataset(
        preprocessed_train,
        train_label,
        mask_train,
        transforms=train_transform,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    testset = ExoDataset(
        preprocessed_test,
        test_label,
        mask_test,
        transforms=train_transform,
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return trainloader, testloader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
