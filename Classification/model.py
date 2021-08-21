#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.modules import conv
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class ConvBlock(): 
    def __init__(self,in_channels,out_channels):
        super(ConvBlock,self).__init__() 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) 
        def forward(self,x):
            return self.conv(x)



class Net(nn.Module):
    def __init__(self,in_channels = 1, out_channels =2 ):
        super(Net,self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels,in_channels*2, 3, stride = 1, padding = 1)
        self.bn1_1 = nn.BatchNorm2d(in_channels*2)
        self.conv1_2 = nn.Conv2d(in_channels*2,in_channels*4,3,stride=1,padding=1)
        self.bn1_2 = nn.BatchNorm2d(in_channels*4)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv2_1 = nn.Conv2d(in_channels*4,in_channels*8, 3, stride = 1, padding = 1)
        self.bn2_1 = nn.BatchNorm2d(in_channels*8)
        self.conv2_2 = nn.Conv2d(in_channels*8,in_channels*16,3,stride=1,padding=1)
        self.bn2_2 = nn.BatchNorm2d(in_channels*16)
        self.maxpoo2_1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3_1 = nn.Conv2d(in_channels*16,in_channels*32, 3, stride = 1, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(in_channels*32)
        self.conv3_2 = nn.Conv2d(in_channels*32,in_channels*64,3,stride=1,padding=1)
        self.bn3_2 = nn.BatchNorm2d(in_channels*64)
        self.maxpoo3_1 = nn.MaxPool2d(kernel_size=2,stride=2)


        #self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride = 1, padding = 1)
        #self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride = 1, padding = 1)

        self.fc1 = nn.Linear(4096,in_channels*64)
        self.fc2 = nn.Linear(in_channels*64,2)

    def forward(self,x):
            #Empieza 1x64x64
        #print(x.shape)
        x = self.conv1_1(x) # num_channels x 64 x 64
        #print(x.shape)
        x = self.bn1_1(x)
        #print(x.shape)
        x = self.conv1_2(x)
        #print(x.shape)
        x = F.relu(self.maxpool_1(x))
        #print(x.shape)
        x = self.conv2_1(x) # num_channels x 32 x 32
        #print(x.shape)
        x = self.bn2_1(x)
        #print(x.shape)
        x = self.conv2_2(x)
        #print(x.shape)
        x = F.relu(self.maxpoo2_1(x))
        #print(x.shape)
        x = self.conv3_1(x) # num_channels x 16 x 16
        #print(x.shape)
        x = self.bn3_1(x)
        #print(x.shape)
        x = self.conv3_2(x)
        #print(x.shape)
        x = F.relu(self.maxpoo3_1(x))
        #print(x.shape)

        x = x.view(-1,64*8*8)
        #print(x.shape)
        #x = F.relu(F.max_pool2d(x, 2)) # num_channels x 32 x 32
        #x = self.conv2(x) # num_channels*2 x 32 x32
        #x = F.relu(F.max_pool2d(x, 2)) #num_channels*2 x 16 x 16
        #x = self.conv3(x) # num_channels*4 x16x16
        #x = F.relu(F.max_pool2d(x, 2)) # num_channels*4 x 8 x 8

        #flatten
        #x = x.view(-1, self.num_channels*4*8*8)

        #fc
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        

        #log_softmax

        x = F.softmax(x,dim=1)
        #print(x.shape)
        #print(x)
        return x





def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert preds.shape == x.shape

