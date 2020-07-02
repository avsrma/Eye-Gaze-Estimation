#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:49:45 2019

@author: avneesh
"""
import torch
import torch.nn as nn

word_to_ix = {"hello": 0, "world": 1}


embeds = nn.Embedding(256, 2)  # 2 words in vocab, 5 dimensional embeddings

poses = torch.LongTensor(128, 2).random_(1, 10)

embed = embeds(poses)
print(embed)