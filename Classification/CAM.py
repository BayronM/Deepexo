#!/usr/bin/env python3

from torchvision import models, transforms
from torch.nn import functional as F
import torch.nn as nn
from torch import topk
import numpy as np
import cv2
import matplotlib.pyplot as plt

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256,256 )
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.bitwise_not(cv2.resize(cam,(width, height)))
        img =  np.tile(255 *orig_image[0][:,:,np.newaxis],3).astype(np.uint8)
        result = heatmap
        plt.imshow(heatmap)
        plt.show()
        plt.imshow(img)
        plt.show()
        extent = 0,64,0,64
        fig = plt.figure(frameon=False)
        im2 = plt.imshow(img,cmap='RdBu_r',interpolation='nearest',extent=extent)
        im1 = plt.imshow(heatmap,alpha=0.7,interpolation='bilinear',extent = extent)
        plt.title(all_classes[class_idx])
        plt.show()
        # put class label text on the result
        #cv2.putText(result, all_classes[class_idx[i]], (20, 40),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        #cv2.imshow('CAM', result/255.)
        #cv2.waitKey(0)
        #cv2.imwrite(f"outputs/CAM_{save_name}.jpg", result)



def CAM(all_classes,model, image):
    classes = all_classes
    features_blobs = []
    model.eval()
    orig_image = image.clone().detach().numpy()
    image_batch = image.unsqueeze(0)
    _,height,width = image.shape
    #print(image_batch.shape)

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    model._modules.get('conv3_2').register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())    
    outputs = model(image_batch)

    probs = F.softmax(outputs,dim=1).data.squeeze()
    class_idx = topk(probs,1)[1].int()
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    # file name to save the resulting CAM image with
    save_name = "cam_0"
    # show and save the results
    show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name)
