# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:14:34 2019

@author: iamav
"""

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
#            nn.Dropout(),
#            nn.Linear(4096, 500),
#            nn.ReLU(inplace=True),
            nn.Linear(4098, 2),
        )

#    def forward(self, x):
#        x = self.features(x)
#        x = self.avgpool(x)
#        x = torch.flatten(x, 1)
#        x = self.classifier(x)
#        return x

    def forward(self, x, y):
        print("x ", x.shape)
        print("y ", y.shape)
        x = self.features(x)
        x = self.avgpool(x)
#        x = torch.model.flatten(x, 1)
        print("x before ",x.shape)                                                                ###flat 64,500
        #x = x.view(x.size(0), -1)                                       ##should be 64,4096 + 2 for y
        #x = self.model.classifier(x)
#        x = self.classifier[2](self.classifier[1](x.view(x.size(0), -1))) #flatten
        x = self.classifier[2](self.classifier[1](torch.flatten(x,1))) #flatten

        print("x flattened ", x.shape)
        x = torch.cat([x, y], dim=1)
        print("x cat ", x.shape)
        x = self.classifier[3](x)
        print("x final ", x.shape)                                      ### 64,2
        return x

m=Model()
print(m)