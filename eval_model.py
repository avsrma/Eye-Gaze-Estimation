# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:26:16 2019

@author: iamav
"""

import torch
import os
import vgg19

outdir = "results/vgg19/00"
model_path = os.path.join(outdir, 'model_state.pth')


model = vgg19.Model()

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
args = checkpoint['args']
optimizer = checkpoint['optimizer']
angle_error = checkpoint['angle_error']
epoch = checkpoint['epoch']

