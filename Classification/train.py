<<<<<<< HEAD
=======
from pickle import FALSE
from numpy.random import f
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
import torch
import functools
from torch._C import device
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter, writer
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Net, VGG_net, YipNet, ResNet50, ResNet101, ResNet152, EfficientNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    show_CAM,
    save_predictions_as_imgs,
)


# Hyperparameters etc.
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
<<<<<<< HEAD
C_RATIO = [0.05, 0.05]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
MODEL = "YipNet"  # Models available "VGG11-13-16-19, NET, YipNet"
VERBOSE = True
=======
C_RATIO = [0.1,0.1]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
MODEL = "ResNet50"       #Models available "VGG11-13-16-19, NET, YipNet"
VERBOSE = True 
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
TRAIN_IMG_DIR = "../data/CNN/master_training.npy"
PSF_IMG_DIR = "../data/CNN/tinyPSF.npy"
TEST_IMG_DIR = "../data/CNN/master_test.npy"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def models_dict():
<<<<<<< HEAD
    dict = {
        "VGG11": VGG_net(type="VGG11"),
        "VGG13": VGG_net(type="VGG13"),
        "VGG16": VGG_net(type="VGG16"),
        "VGG19": VGG_net(type="VGG19"),
        "NET": Net(),
        "YipNet": YipNet(),
        "ResNet50": ResNet50(),
    }

    return dict


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
=======
    dict =  {
        'VGG11' : VGG_net(type='VGG11'),
        'VGG13' : VGG_net(type='VGG13'),
        'VGG16' : VGG_net(type='VGG16'),
        'VGG19' : VGG_net(type='VGG19'),
        'NET' : Net(),
        'YipNet' : YipNet(),
        'ResNet50': ResNet50(),
    }

    return dict

>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911

        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tdqm loop
        loop.set_postfix(loss=loss.item())
    return loss.item()

def train_fn(loader, model , optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

<<<<<<< HEAD
def main():
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(np.random.randint(0, 359)),
        ]
    )
=======
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
    return loss.item()
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911

    models = models_dict()

    print(f"\nYour device is {DEVICE} and the model is {MODEL}\n\n")

    model = models.get(MODEL)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

<<<<<<< HEAD
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch = 0
=======
    models = models_dict()

    print(f"\nYour device is {DEVICE} and the model is {MODEL}\n\n")

    model = models.get(MODEL)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model.to(DEVICE)
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch = 0    
    
    trainloader, testloader = get_loaders(
        TRAIN_IMG_DIR,
        PSF_IMG_DIR,
        C_RATIO,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        num_workers=4,
<<<<<<< HEAD
        pin_memory=True,
    )

    if LOAD_MODEL:
        checkpoint = torch.load("checkpoint_YipNet--.pth")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"\n\nStarting from epoch - {epoch} - \n\n")

    print(model)
    # check_accuracy(testloader,model,loss_fn,device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epoch, NUM_EPOCHS):
        print(f'{"-"*20}\nEpoch {epoch + 1 } | {NUM_EPOCHS}\n')
        loss = train_fn(trainloader, model, optimizer, loss_fn, scaler)
        check_accuracy(testloader, model, loss_fn, device=DEVICE)

    # writer = SummaryWriter('runs/Experiment_1')

    if SAVE_MODEL:
=======
        pin_memory=True)
    
    if LOAD_MODEL:
        checkpoint = torch.load("checkpoint_YipNet--.pth")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"\n\nStarting from epoch - {epoch} - \n\n")

    print(model)
    #check_accuracy(testloader,model,loss_fn,device=DEVICE)
    scaler= torch.cuda.amp.GradScaler()

    for epoch  in range(epoch,NUM_EPOCHS):
        print(f'{"-"*20}\nEpoch {epoch + 1 } | {NUM_EPOCHS}\n')
        loss=train_fn(trainloader,model,optimizer,loss_fn,scaler)
        check_accuracy(testloader,model,loss_fn,device=DEVICE)
    
    #writer = SummaryWriter('runs/Experiment_1')

    if SAVE_MODEL: 
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
<<<<<<< HEAD
            "epoch": epoch + 1,
        }
        save_checkpoint(
            checkpoint, f"model_{MODEL}_{epoch+1}_[{C_RATIO[0],C_RATIO[1]}].pth"
        )


if __name__ == "__main__":
=======
            "epoch" : epoch + 1 
        }
        save_checkpoint(checkpoint,f"model_{MODEL}_{epoch+1}_[{C_RATIO[0],C_RATIO[1]}].pth")


    
    
if __name__ == "__main__" :
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
    main()
