from pickle import FALSE
from unicodedata import name
from numpy.random import f
import torch
import functools
from torch._C import device
from torchvision import transforms
from tqdm import tqdm
from dataset import ExoDataset, local_normal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Net,VGG_net, YipNet
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
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
C_RATIO = [0.1,0.1]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
MODEL = "Net"       #Models available "VGG11-13-16-19, NET, YipNet"
VERBOSE = True 
TRAIN_IMG_DIR = "../data/CNN/master_training.npy"
PSF_IMG_DIR = "../data/CNN/tinyPSF.npy"
TEST_IMG_DIR = "../data/CNN/master_test.npy"
VAL_IMG_DIR = "../data/GAN/Real_test_confirmed.npy"



def models_dict():
    dict =  {
        'VGG11' : VGG_net(type='VGG11'),
        'VGG13' : VGG_net(type='VGG13'),
        'VGG16' : VGG_net(type='VGG16'),
        'VGG19' : VGG_net(type='VGG19'),
        'Net' : Net(),
        'YipNet' : YipNet()
    }

    return dict




def train_fn(loader, model , optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

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



def main():

    models = models_dict()

    print(f"\nYour device is {DEVICE} and the model is {MODEL}\n\n")

    model = models.get(MODEL)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model.to(DEVICE)

    c_ratio = [0.8,1]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler= torch.cuda.amp.GradScaler()
    epoch = 0  

    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(np.random.randint(0,359))
    ]) 
    val_transform = transforms.Compose([
    transforms.ToTensor()
    ])


    valdata = np.load(VAL_IMG_DIR)
    valdata = local_normal(valdata)
    val_label = [1] * valdata.shape[0]
    valset = ExoDataset(valdata,val_label,[],val_transform)
    valloader = DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    accuaracy = 0
    acc = 0
    val_accuaracy = []
    while epoch < 1000: 
    
        trainloader, testloader = get_loaders(
            TRAIN_IMG_DIR,
            PSF_IMG_DIR,
            c_ratio,
            TEST_IMG_DIR,
            BATCH_SIZE,
            train_transform,
            num_workers=4,
            pin_memory=True)

        if (check_accuracy(testloader,model,loss_fn,device=DEVICE) < 0.85):
            while accuaracy < 0.85 :
                print(f'{"-"*20}\nEpoch {epoch + 1 } | {c_ratio}\n')
                loss=train_fn(trainloader,model,optimizer,loss_fn,scaler)
                accuaracy = check_accuracy(testloader,model,loss_fn,device=DEVICE)
                if accuaracy <= (acc*0.8):
                    checkpoint=torch.load(name)
                    model.load_state_dict(checkpoint['state_dict'])
                acc = accuaracy
                epoch += 1
        name = f"model_{MODEL}_{epoch+1}_[{c_ratio[0],c_ratio[1]}].pth"
        if SAVE_MODEL: 
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
                "epoch" : epoch + 1 
            }
            save_checkpoint(checkpoint,name)

        c_ratio[0] = c_ratio[0] * 0.9
        c_ratio[1] = c_ratio[1] * 0.9 
        accuaracy = 0
        val_accuaracy.append(check_accuracy(valloader,model,loss_fn,DEVICE))  
    
    
if __name__ == "__main__" :
    main()
