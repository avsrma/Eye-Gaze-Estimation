# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:08:04 2019

@author: iamav
"""

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        vgg_firstlayer=models.vgg19(pretrained = True).features[0] #load just the first conv layer
        vgg_1=models.vgg19(pretrained = True).features[1:14] #load upto the classification layers except first conv layer
        vgg_2=models.vgg19(pretrained = True).features[14:25]
        vgg_3=models.vgg19(pretrained = True).features[25:32]
        vgg_4=models.vgg19(pretrained = True).features[32:36]
        
        w1=vgg_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=vgg_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=vgg_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional
    
        first_conv=nn.Conv2d(1, 64, 3, padding = (1,1)) #create a new conv layer
        first_conv.weight=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias
    
    
        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.vgg_1 =nn.Sequential(vgg_1)
        self.dropout_1 = nn.Dropout(0.5, inplace=True)
        self.vgg_2 =nn.Sequential(vgg_2)
        self.dropout_2 = nn.Dropout(0.5, inplace=True)
        self.vgg_3 =nn.Sequential(vgg_3)
        self.dropout_3 = nn.Dropout(0.5, inplace=True)
        self.vgg_4 =nn.Sequential(vgg_4)
        
        
        self.fc1 = nn.Linear(3072, 1000)
        self.fc2 = nn.Linear(1002, 2)

    def forward(self, x, y):
        x=self.first_convlayer(x)
        x=self.vgg_1(x)        
        x=self.vgg_2(x)
        x=self.vgg_3(x)
        x=self.vgg_4(x)
        
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)

        return x

#print(m.fc1.weight)
m =Model()
print(m)