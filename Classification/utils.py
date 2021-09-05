#!/usr/bin/env python3

import torch
import torchvision
import numpy as np
from dataset import ExoDataset, ExoDatasetMask, data_preprocess
from torch.utils.data import DataLoader
from CAM import CAM
import matplotlib.pyplot as plt

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
    print(train_label.shape)
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

def get_loaders_m(
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
    print(train_label.shape)
    preprocessed_test, test_label, mask_test =  data_preprocess(np.load(test_images),
                                                                np.load(psf_images),
                                                                c_ratio=c_ratio,
                                                                no_blend = True)
    trainset = ExoDatasetMask(
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

    testset = ExoDatasetMask(
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

def check_accuracy(loader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    accuaracy = correct/len(loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * accuaracy ))
    return(accuaracy)

def check_accuracy_m(loader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target,masks in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    accuaracy = correct/len(loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * accuaracy ))
    return(accuaracy)

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

def show_CAM(dataloader, model, n_examples, classes,loc = False):
    """"
    plot a class activation for a dataloader use a CAM.py file
    !! only for testing purpouses
    n_examples : number of examples to show (need to be <= batch_size)
    """
    for i in range(n_examples):
        (images,labels,masks) = next(iter(dataloader))
        plt.imshow(masks[i])
        plt.show()
        CAM(classes,model.cpu(),images[i].cpu())