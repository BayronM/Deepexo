
from cv2 import GaussianBlur
from torch.utils.tensorboard import SummaryWriter, writer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import local_normal
from torchvision.transforms.transforms import Grayscale
from dataset import ExoDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Net,VGG_net, YipNet
from utils import (
    get_loaders_m,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    check_accuracy_m,
    show_CAM,
    save_predictions_as_imgs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
BATCH_SIZE = 32
C_RATIO = [0.1,0.1]
MODEL = "NET"
MODEL_PATH = "./model_Net_584_[(0.030521633958155707, 0.03815204244769462)].pth"
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
        'NET' : Net(),
        'YipNet' : YipNet()
    }

    return dict



def main():
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(np.random.randint(0,359))
    ])

    val_transform = transforms.Compose([
    transforms.ToTensor()
    ])

    trainloader, testloader = get_loaders_m(
        TRAIN_IMG_DIR,
        PSF_IMG_DIR,
        C_RATIO,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        num_workers=4,
        pin_memory=True)

    models = models_dict()

    print(f"\nYour device is {DEVICE.upper()} and the model is {MODEL}\n\n")
    classes = ['no_planet', 'planet']
    model = models.get(MODEL)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    print(f"\n\nTrain with  {epoch} - epochs - \n\n")

    print("--------------------------------")
    print(f"\n Test Set :\n")
    check_accuracy_m(testloader, model, loss_fn, DEVICE)
    show_CAM(testloader,model,10,classes)

#------------------------------------------------------------------
    # Test Validation Set


    valdata = np.load(VAL_IMG_DIR)
    valdata = local_normal(valdata)
    val_label = [1] * valdata.shape[0]
    valset = ExoDataset(valdata,val_label,[],val_transform)
    valloader = DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("--------------------------------")
    print(f"\n Validation Set :\n")
    check_accuracy(valloader, model.to(DEVICE), loss_fn, DEVICE)

    show_CAM(valloader,model,10,classes,loc = False)


#-----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
