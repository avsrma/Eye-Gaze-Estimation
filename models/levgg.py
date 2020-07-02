# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:58:57 2019

@author: iamav
"""


import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class Upsampling(nn.Module):
    def __init__(self):
        super(Upsampling, self).__init__()
        ### upsample
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.upsam = nn.Upsample(scale_factor=(60,2))

    def forward(self, y):
        y = y.unsqueeze_(0)
        y = y.reshape(-1,1,2)

        y = self.conv1(y)
        y = self.conv2(y)
        y = y.unsqueeze_(0)
        y = y.reshape(-1,1,64,2)
        y = self.upsam(y)

        return y

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)

class PosNet(nn.Module):
    def __init__(self):
        super(PosNet, self).__init__()

        self.upsampling = upsampling

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        
        self.fc1 = nn.Linear(15000, 4096)  
        self.fc2 = nn.Linear(4096, 1024)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x, y):
        y = self.upsampling(y)
        x = x.view(-1, 36, 60).bmm(y.view(-1, 60, 256))
        x = x.unsqueeze_(0).reshape(-1,1,36,256)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

#        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
#        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        
#        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
#        x = self.fc2(x)
        return x

class ImNet(nn.Module):
    def __init__(self):
        super(ImNet, self).__init__()

        first_conv=nn.Conv2d(1, 64, 1) #create a new conv layer
        vgg=models.vgg19(pretrained = True).features[1:37] #load upto the classification layers except first conv layer


        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.vgg = vgg

        self.avgpool = models.vgg19(pretrained=True).avgpool

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 1024)    

    def forward(self, x):
        x=self.first_convlayer(x)
        x=self.vgg(x)
        x = self.avgpool(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = self.fc2(x)
#        print('imnet', x.shape)
        return x


class PosUp(nn.Module):
    def __init__(self):
        super(PosUp, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 1024)

    def forward(self, y):
        y=y.reshape(-1,1,2)
        y = self.fc1(y.view(y.size(0), -1)) #flatten
        y=self.fc2(y)
        return y


class Ensemble(nn.Module):
    def __init__(self, imnet, posnet, posup):
        super(Ensemble, self).__init__()
        self.ImNet = imnet
        self.PosNet = posnet
        self.PosUp = posup

        self.classifier = nn.Sequential(nn.Linear(3072, 512),
                                        nn.ReLU(True), 
                                        nn.Dropout(0.5),
                                        nn.Linear(512, 128),
                                        nn.ReLU(True), 
                                        nn.Dropout(0.5),
                                        nn.Linear(128, 2)
                                        )
        
    def forward(self, x, y):
        x1 = self.ImNet(x)
        x2 = self.PosNet(x, y)
        x3 = self.PosUp(y)
        
#        print('ensemble', x1.shape, x2.shape, x3.shape)
        x = torch.cat((x1, x2, x3), dim=1)
#        print
        x = self.classifier(x)
        return x


upsampling = Upsampling()
posup = PosUp()
posnet = PosNet()
imnet = ImNet()

model = Ensemble(imnet, posnet, posup)
print(model.parameters)
