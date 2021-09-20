<<<<<<< HEAD
# Train version for data collecting


import torch
=======
from pickle import FALSE
from unicodedata import name
from numpy.random import f
import torch
import functools
from torch._C import device
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
from torchvision import transforms
from tqdm import tqdm
from dataset import ExoDataset, local_normal
from torch.utils.data import DataLoader
<<<<<<< HEAD
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import Net, VGG_net, YipNet
from utils import (
    save_checkpoint,
    get_loaders,
    check_accuracy,
=======
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
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
)


# Hyperparameters etc.
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
<<<<<<< HEAD
C_RATIO = [0.1, 0.1]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
MODEL = "YipNet"  # Models available "VGG11-13-16-19, NET, YipNet"
VERBOSE = True
=======
C_RATIO = [0.1,0.1]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
MODEL = "Net"       #Models available "VGG11-13-16-19, NET, YipNet"
VERBOSE = True 
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
TRAIN_IMG_DIR = "../data/CNN/master_training.npy"
PSF_IMG_DIR = "../data/CNN/tinyPSF.npy"
TEST_IMG_DIR = "../data/CNN/master_test.npy"
VAL_IMG_DIR = "../data/GAN/Real_test_confirmed.npy"
<<<<<<< HEAD
SAVE_TO_TEX = True
TEXNAME = "./Tex/YipNet.tex"


def models_dict():
    dict = {
        "VGG11": VGG_net(type="VGG11"),
        "VGG13": VGG_net(type="VGG13"),
        "VGG16": VGG_net(type="VGG16"),
        "VGG19": VGG_net(type="VGG19"),
        "Net": Net(),
        "YipNet": YipNet(),
=======



def models_dict():
    dict =  {
        'VGG11' : VGG_net(type='VGG11'),
        'VGG13' : VGG_net(type='VGG13'),
        'VGG16' : VGG_net(type='VGG16'),
        'VGG19' : VGG_net(type='VGG19'),
        'Net' : Net(),
        'YipNet' : YipNet()
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
    }

    return dict


<<<<<<< HEAD
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backwards
=======


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
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

<<<<<<< HEAD
        # update tdqm loop
=======
        #update tdqm loop
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
        loop.set_postfix(loss=loss.item())
    return loss.item()


<<<<<<< HEAD
def main():

    if SAVE_TO_TEX:
        infile = open("./Tex/table_i.tex", "r")
        outfile = open(TEXNAME, "w")
        inlines = infile.readlines()
        outfile.writelines(inlines)
        infile.close()
=======

def main():
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911

    models = models_dict()

    print(f"\nYour device is {DEVICE} and the model is {MODEL}\n\n")

    model = models.get(MODEL)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
<<<<<<< HEAD
    model.to(DEVICE)

    c_ratio = [0.8, 1]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    epoch = 0

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(np.random.randint(0, 359)),
        ]
    )
    val_transform = transforms.Compose([transforms.ToTensor()])
=======
    
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

>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911

    valdata = np.load(VAL_IMG_DIR)
    valdata = local_normal(valdata)
    val_label = [1] * valdata.shape[0]
<<<<<<< HEAD
    valset = ExoDataset(valdata, val_label, [], val_transform)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False,)
    accuaracy = 0
    acc = 0
    val_accuaracy = []
    name = ""
    while epoch < 1000:
=======
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
    
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
        trainloader, testloader = get_loaders(
            TRAIN_IMG_DIR,
            PSF_IMG_DIR,
            c_ratio,
            TEST_IMG_DIR,
            BATCH_SIZE,
            train_transform,
            num_workers=4,
<<<<<<< HEAD
            pin_memory=True,
        )

        if check_accuracy(testloader, model, loss_fn, device=DEVICE) < 0.85:
            while accuaracy < 0.85:
                print(f'{"-"*20}\nEpoch {epoch + 1 } | {c_ratio}\n')
                loss = train_fn(trainloader, model, optimizer, loss_fn, scaler)
                accuaracy = check_accuracy(testloader, model, loss_fn, device=DEVICE)
                if accuaracy <= (acc * 0.8):
                    checkpoint = torch.load(name)
                    model.load_state_dict(checkpoint["state_dict"])
                acc = accuaracy
                epoch += 1
        name = f"Models/model_{MODEL}_{epoch+1}_[{c_ratio[0],c_ratio[1]}].pth"
        if SAVE_MODEL:
=======
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
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
<<<<<<< HEAD
                "epoch": epoch + 1,
            }
            save_checkpoint(checkpoint, name)
        if SAVE_TO_TEX:
            line = f"{epoch}&{loss}&{c_ratio}&{accuaracy} \\"
            outfile.write(line)

        c_ratio[0] = c_ratio[0] * 0.9
        c_ratio[1] = c_ratio[1] * 0.9
        accuaracy = 0
        val_accuaracy.append(check_accuracy(valloader, model, loss_fn, DEVICE))


if __name__ == "__main__":
=======
                "epoch" : epoch + 1 
            }
            save_checkpoint(checkpoint,name)

        c_ratio[0] = c_ratio[0] * 0.9
        c_ratio[1] = c_ratio[1] * 0.9 
        accuaracy = 0
        val_accuaracy.append(check_accuracy(valloader,model,loss_fn,DEVICE))  
    
    
if __name__ == "__main__" :
>>>>>>> aa04b5d0fcdeefbacd866a4d855a37ca4a385911
    main()
