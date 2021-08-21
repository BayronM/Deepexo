#!/usr/bin/env python3

import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from model import UNET , Net
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
)
from CAM import CAM


# Hyperparameters etc.
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
C_RATIO = [0.01,0.5]
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../data/CNN/master_training.npy"
PSF_IMG_DIR = "../data/CNN/tinyPSF.npy"
TEST_IMG_DIR = "../data/CNN/master_test.npy"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"



def train(dataloader,model,loss_fn,optimizer,epoch):
    train_losses = []
    train_counter = []
    loop = tqdm(dataloader)
    model.train()
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.to(device=DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss  = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(dataloader.dataset),
            #    100. * batch_idx / len(dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(dataloader.dataset)))
            torch.save(model.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device=DEVICE)
            target = target.to(device=DEVICE)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    #test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))







def main():
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(np.random.randint(0,359))
    ])


    model = Net().to(DEVICE)
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
        load_checkpoint(torch.load("model.pth"),model)
    check_accuracy(testloader,model,loss_fn)
    #scaler= torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f'{"-"*20}\nEpoch {epoch + 1 } | {NUM_EPOCHS}\n')
        train(trainloader,model,loss_fn,optimazer,epoch)
        check_accuracy(testloader,model,loss_fn)

    #classes = ['planet','no planet']
    #for i in range(15):
    #    (images,labels) = next(iter(testloader))
    #    CAM(classes,model.cpu(),images[i].cpu())



if __name__ == "__main__" :
    main()
