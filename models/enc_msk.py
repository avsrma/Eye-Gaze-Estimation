# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:14:08 2019

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
        self.conv1 = nn.Conv1d(1, 64, 1)
        self.upsam = nn.Upsample(scale_factor=(60,2))
        
    def forward(self, y):
        y = y.unsqueeze_(0) 
        #print('up y', y.shape)
        y = y.reshape(-1,1,2)
        
        y = self.conv1(y)
        y = y.unsqueeze_(0)
        y = y.reshape(-1,1,64,2)
        y = self.upsam(y)
        #print(y.shape)

        return y


class PosNet(nn.Module):
    def __init__(self):
        super(PosNet, self).__init__()
        
        self.upsampling = upsampling
        
        first_conv = nn.Conv2d(1, 64, 3) #create a new conv layer
        vgg = models.vgg19(pretrained = True).features[2:37] #load upto the classification layers except first conv layer
    
        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.vgg = vgg
        
        self.avgpool = models.vgg19(pretrained=True).avgpool

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        
        self.fine_tune()

    def forward(self, x, y):
        y = self.upsampling(y)
        #print(x.shape)
        #print(y.shape)
        x = x.view(-1, 36, 60).bmm(y.view(-1, 60, 256))
        x = x.unsqueeze_(0).reshape(-1,1,36,256)
        #print(x.shape)

        x = self.first_convlayer(x) 
        #print('before ',x.shape)
        x = self.vgg(x)
        #print('after ', x.shape)
        x = self.avgpool(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = self.fc2(x)

        return x
    
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.vgg.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.vgg)[5:31]:
            for p in c.parameters():
                p.requires_grad = fine_tune



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

        self.fine_tune()
        
    def forward(self, x):
        x=self.first_convlayer(x)
        x=self.vgg(x)
        x = self.avgpool(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
#        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)

        return x
    
    def fine_tune(self, fine_tune=True):
            """
            Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
            :param fine_tune: Allow?
            """
            for p in self.vgg.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.vgg)[5:31]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

class Ensemble(nn.Module):
    def __init__(self, imnet, posnet):
        super(Ensemble, self).__init__()
        self.ImNet = imnet
        self.PosNet = posnet
        
        self.classifier = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(True), 
                                        nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(True), 
                                        nn.Dropout(0.5),
                                        nn.Linear(512, 2)
                                 )
    def forward(self, x, y):
        x1 = self.ImNet(x)
        x2 = self.PosNet(x, y)
        x = torch.cat((x1, x2), dim=1)

        x = self.classifier(x)
        #x = x.view(x.size(0), -1) #flatten     
        #y = y.reshape(-1,2)
        #x = torch.cat([x, y], dim=1)
#        print("cls(relu x )", x.shape)
#        x = torch.cat([x,y], dim=1)
#        print("final cat xy", x.shape)
        return x


#x = torch.randn(128, 2)
#upsampling = Upsampling()


upsampling = Upsampling()
posnet = PosNet()
imnet = ImNet()

model = Ensemble(imnet, posnet)
print(model.parameters)
























#def Upscale(x):

#conv = nn.Conv1d(1, 64, 1)
#conv2d = nn.Conv2d(1, 128, 1)
#up = nn.UpsamplingBilinear2d(size=(60,14))
#
#
##x = x.unsqueeze_(0)
#x = torch.randn(128,2)
#x = x.reshape(128, 1, 2)

#print(x.shape)
#
#x = conv(x)
##x = x.view(1,1,128,2)
#x = x.unsqueeze_(0)
#x =  x.reshape(128,1,64,2)
##print(x.shape)
#
#x = up(x)
#print(x.shape)
##x = x.view(x.size(0), -1)
##x = fc(x)
#
##out = conv2d(x)
##print(out)
#
#
#y = torch.randn(128,1, 36,60)
#y = y.reshape(128,1,36,60)
#
#print(y.shape)
#
#xy = y.view(128, -1, 60).bmm(x.view(128, 60,-1))
#xy = xy.unsqueeze_(0).reshape(128,1,36,14)
#
#print(xy.shape)











