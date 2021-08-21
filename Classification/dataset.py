#!/usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms


#print(f"Shape of psf data: {psf_images.shape}")
#print(f"Shape of test data: {test_images.shape}")
#print(f"Shape of validation data: {validation_images.shape}")

def local_normal(data):
    new_imgs_list = []
    for imgs in data:
        local_min = np.min(imgs)
        new_imgs = (imgs - local_min) / np.max(imgs - local_min)
        new_imgs_list.append(new_imgs)
    return np.array(new_imgs_list).reshape(-1, 64, 64)

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx]

def inject_planet(data, psf_library, c_ratio=[0.01, 0.1],
                  x_bound=[4, 61], y_bound=[4, 61], no_blend=False):
    """Inject planet into random location within a frame
    data: single image
    psf_library: collection of libarary (7x7)
    c_ratio: the contrast ratio between max(speckle) and max(psf)*, currently accepting a range
    x_bound: boundary of x position of the injected psf, must be within [0,64-7]
    y_bound: boundary of y position of the injected psf, must be within [0,64-7]
    no_blend: optional flag, used to control whether two psfs can blend into each other or not, default option allows blending.
    """

    image = data.copy()
    pl_num = np.random.randint(1, high=4)
    pos_label = np.zeros([64, 64])
    used_xy = np.array([])
    c_prior = np.linspace(c_ratio[0], c_ratio[1], 100)
    if x_bound[0] < 4 or x_bound[0] > 61:
        raise Exception("current method only injects whole psf")
    if y_bound[0] < 4 or y_bound[0] > 61:
        raise Exception("current method only injects whole psf")

    for num in range(pl_num):
        while True:
            psf_idx = np.random.randint(0, high=psf_library.shape[0])
            Nx = np.random.randint(x_bound[0], high=x_bound[1])
            Ny = np.random.randint(y_bound[0], high=y_bound[1])
            if len(used_xy) == 0:
                pass
            else:
                if no_blend:
                    if np.any(np.linalg.norm(np.array([Nx, Ny]) - used_xy) < 3):
                        pass
                else:
                    if np.any(np.array([Nx, Ny]) == used_xy):
                        pass
            if np.linalg.norm(np.array([Nx, Ny]) - np.array([32.5, 32.5])) < 4:
                pass
            else:
                planet_psf=crop_center(psf_library,7,7)
                brightness_f = c_prior[0] * np.max(image) / np.max(planet_psf)
                mod = planet_psf * brightness_f
                image[Ny - 4:Ny + 3, Nx - 4:Nx + 3] += mod
                used_xy = np.append(used_xy, [Nx, Ny]).reshape(-1, 2)
                pos_label[Ny - 4:Ny + 3, Nx - 4:Nx + 3] = 1
                break
    return image, pos_label

def data_preprocess(data,psf_pl,c_ratio,no_blend):

    ## inject planet for train_data
    injected_samples = np.zeros([len(data), 64, 64])
    planet_loc_maps = np.zeros([len(data)*2, 64, 64])
    for i in range(len(data)):
        new_img, loc_map = inject_planet(data[i].reshape(64, 64), psf_pl, c_ratio=c_ratio,no_blend=True)
        injected_samples[i] += new_img
        planet_loc_maps[i] += loc_map

    normalised_injected = local_normal(injected_samples)
    nor_data = local_normal(data)

    dataset = np.zeros([int(len(data) * 2), 64, 64])

    ## Here we normalised each images into [0,1]
    dataset[:len(data)] += normalised_injected
    dataset[len(data):] += nor_data

    label = np.zeros((len(dataset)))
    label[:len(data)] += 1

    print("label size =", label.shape)
    print("data size=", dataset.shape)
    print("number of positive examples", np.sum(label))

    return dataset, label, planet_loc_maps



class ExoDataset(Dataset):
    def __init__(self,files,labels,masks, transforms=None):
        self.files = files
        self.labels = labels
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        image =np.float32(np.moveaxis(self.files[index],-1,0))
        if self.transforms:
            image = self.transforms(image)
        return image,int(self.labels[index])

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(np.random.randint(0,359))
    ])
