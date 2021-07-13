#!/usr/bin/env python3

import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
C_RATIO = [1,1000]
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/data/CNN/master_training.npy"
PSF_IMG_DIR = "/data/CNN/tinyPSF.npy"
TEST_IMG_DIR = "/data/CNN/master_test.npy"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def train_fn(loader, model , optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions,targets)

        #backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tdqm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(np.random.randint(0,359))
    ])


    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimazer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainloader, testloader = get_loaders(
        TRAIN_IMG_DIR,
        PSF_IMG_DIR,
        C_RATIO,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        num_workers=4,
        pin_memory=True)

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint.pth"),model)

    check_accuracy(testloader,model,device=DEVICE)
    scaler= torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(trainloader,model,optimazer,loss_fn,scaler)

    #save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimazer.state_dict()
    }
    save_checkpoint(checkpoint)


    #check accuaracy
    check_accuracy(testloader,model,device=DEVICE)

    save_predictions_as_imgs(testloader,model,folder ="predict_images/",device=DEVICE)


if __name__ == "__main__" :
    main()
