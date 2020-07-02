# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 20:01:00 2019

@author: iamav
"""

# coding: utf-8

import os
import numpy as np

import torch
import torchvision
import argparse

import torch.utils.data
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch', type=str, required=True, choices=['lenet', 'resnet_preact', 'izenet'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--test_id', type=int, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)

    # optimizer
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', type=str, default='[20, 30]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    return args

class MPIIGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        path = os.path.join(dataset_dir, '{}.npz'.format(subject_id))
        with np.load(path) as fin:
            self.images = fin['image']
            self.poses = fin['pose']
            self.gazes = fin['gaze']
#            self.heads  = fin['heads']
        self.length = len(self.images)

        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.poses = torch.from_numpy(self.poses)
        self.gazes = torch.from_numpy(self.gazes)
#        self.heads = torch.from_numpy(self.heads)
    def __getitem__(self, index):
        return self.images[index], self.poses[index], self.gazes[index], self.heads[index]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__
    

def get_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)
    subject_ids = ['p{:02}'.format(index) for index in range(15)]
    test_subject_id = subject_ids[test_subject_id]

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
        if subject_id != test_subject_id
    ])
    test_dataset = MPIIGazeDataset(test_subject_id, dataset_dir)

    assert len(train_dataset) == 42000
    assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    
    return test_loader

def test(test_loader):
#    test_loader = get_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu)    
    for step, (images) in enumerate(test_loader):
        print(images.size)
        img = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
    return img


args = parse_args()
test_loader = get_loader(
    args.dataset, args.test_id, args.batch_size, args.num_workers, True)

test(test_loader)
