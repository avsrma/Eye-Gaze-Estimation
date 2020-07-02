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
        vgg_firstlayer=models.vgg16(pretrained = True).features[0] #load just the first conv layer
        vgg=models.vgg16(pretrained = True).features[1:30] #load upto the classification layers except first conv layer

        w1=vgg_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=vgg_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=vgg_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional
    
        first_conv=nn.Conv2d(1, 64, 3, padding = (1,1)) #create a new conv layer
        first_conv.weigth=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias
    
    
        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.vgg =nn.Sequential(vgg)

        self.fc1 = nn.Linear(3072, 1000)
        self.fc2 = nn.Linear(1002, 2)

    def forward(self, x, y):
#        print("x ", x.shape)
#        print("y ", y.shape)
        x=self.first_convlayer(x)
        x=self.vgg(x)
#        print("x before ",x.shape)                                                                ###flat 64,500
#        x = x.view(x.size(0), -1)                                       ##should be 64,4096 + 2 for y
        #x = self.model.classifier(x)
#        x = self.classifier[2](self.classifier[1](x.view(x.size(0), -1))) #flatten
#        x = self.classifier[2](self.classifier[1](torch.flatten(x,1))) #flatten
#        x=x.view(-1, 7*7*512)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
#        print("x relu flat ", x.shape)
        x = torch.cat([x, y], dim=1)
#        print("cat x", x.shape)
#        print("cat y", y.shape)
        
        x = self.fc2(x)
#         print("x final ", x.shape)                                      ### 64,2
        return x
