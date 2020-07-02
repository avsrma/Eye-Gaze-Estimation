#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:16:05 2019

@author: avneesh
"""

        


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import numpy as np
#from torchviz import make_dot

class VGG_ATT(nn.Module):
    def __init__(self, mode='pc'):
        super(VGG_ATT, self).__init__()
        self.mode = mode

        self.features = models.vgg19(pretrained = True).features[0:36]  
#        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(1,1)

        self.classifier = nn.Linear(512, 10)
        
#        self.conv_att = nn.Conv2d(2,64,3)
        self.l1 = nn.Sequential(*list(self.features)[:17],
                                            #nn.BatchNorm2d(256),
                                            nn.ReLU(True))
                                            #nn.MaxPool2d(2,2))
        #self.maxpool2 = nn.MaxPool2d(2,2)
        
        self.l2 = nn.Sequential(*list(self.features)[19:24],
                                           # nn.BatchNorm2d(512),
                                            nn.ReLU(True))
                                            #nn.MaxPool2d(2,2))
        #self.bn1 = nn.BatchNorm2d(512)
        #self.r2 = nn.ReLU(True)
        #self.m2 = nn.MaxPool2d(2,2)
        


        self.l3 = nn.Sequential(*list(self.features)[28:29],
                                      #nn.BatchNorm2d(512),
                                      nn.ReLU(True),
                                      *list(self.features)[32:33],
                                      #nn.BatchNorm2d(512),
                                      nn.ReLU(True),
                                      *list(self.features)[32:33],
                                      #nn.BatchNorm2d(512),
                                      nn.ReLU(True)
                                      #nn.MaxPool2d(2,2)
                                      )
#        self.bn2 = nn.BatchNorm2d(512)
#        self.r3 = nn.ReLU(True)
#        self.m3 = nn.MaxPool2d(2,2)

        if mode == 'pc':
            self.u1 = nn.Conv2d(256, 1, 1)
            self.u2 = nn.Conv2d(512, 1, 1)
            self.u3 = nn.Conv2d(512, 1, 1)




        self.conv_out = nn.Sequential(*list(self.features)[32:33],
                                      #nn.BatchNorm2d(512),
                                      nn.ReLU(True),
                                      nn.MaxPool2d(2,2),
                                      *list(self.features[34:35]),
                                      #nn.BatchNorm2d(512),
                                      nn.ReLU(True))
                                      #nn.MaxPool2d(2,2))
#        self.bn3 = nn.BatchNorm2d(512)
#        self.r4 = nn.ReLU(True)
#        self.m4 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(14336, 512)

        self.fc1_l1 = nn.Linear(512, 256)
        self.fc1_l2 = nn.Linear(512, 512)
        self.fc1_l3 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(256 + 512 + 512, 500)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)

        conv_out = self.conv_out(l3)
        fc1 = self.fc1(conv_out.view(conv_out.size(0), -1))
        fc1_l1 = self.fc1_l1(fc1)
        fc1_l2 = self.fc1_l2(fc1)
        fc1_l3 = self.fc1_l3(fc1)

        att1 = self._compatibility_fn(l1, fc1_l1, level=1)
        att2 = self._compatibility_fn(l2, fc1_l2, level=2)
        att3 = self._compatibility_fn(l3, fc1_l3, level=3)

        g1 = self._weighted_combine(l1, att1)
        g2 = self._weighted_combine(l2, att2)
        g3 = self._weighted_combine(l3, att3)

        g = torch.cat((g1, g2, g3), dim=1)
        out = self.fc2(g)

        return out

    def _compatibility_fn(self, l, g, level):
        if self.mode == 'dp':
            att = l * g.unsqueeze(2).unsqueeze(3)
            att = att.sum(1).unsqueeze(1)

            size = att.size()
            att = att.view(att.size(0), att.size(1), -1)
            att = F.softmax(att, dim=2)
            att = att.view(size)
        elif self.mode == 'pc':
            att = l + g.unsqueeze(2).unsqueeze(3)

            if level == 1:
                u = self.u1
            elif level == 2:
                u = self.u2
            elif level == 3:
                u = self.u3
            att = u(att)
            
            size = att.size()
            att = att.view(att.size(0), att.size(1), -1)
            att = F.softmax(att, dim=2)
            att = att.view(size)

        return att
    
    def _weighted_combine(self, l, att_map):
        g = l * att_map

        return g.view(g.size(0), g.size(1), -1).sum(2)


class Model(nn.Module):
    def __init__(self, VGG_ATT):
        super(Model, self).__init__()
        self.first_conv=nn.Conv2d(1, 3, 3, padding = (1,1)) #create a new conv layer
        self.VGG_ATT = VGG_ATT
        self.fc3 = nn.Linear(502, 2)
        
    def forward(self, x, y):
        x = self.first_conv(x)
        x = self.VGG_ATT(x)
        x = F.relu(x.view(x.size(0), -1), inplace=True) #flatten
        x = torch.cat([x, y], dim=1)
        x = self.fc3(x)

        return x
        
vgg_att = VGG_ATT()
m = Model(vgg_att)
print(m)
