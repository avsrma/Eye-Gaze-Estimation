#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:07:09 2019

@author: avneesh
"""

#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
from collections import OrderedDict
import importlib
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torchvision
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch', type=str, required=True, choices=['lenet', 'resnet_preact', 'vgg16', 'vgg19', 'resnet50', 'inception', 'ensemble', 'vgg_att', 'vgg_att_prt', 'ens-att'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--test_id', type=int, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)

    # optimizer
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[15, 25]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--cooldown', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-8)
#    parser.add_argument('--eps', type=float, default=1e-8)
    
    # TensorBoard
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false', default=True)
    parser.add_argument('--tensorboard_images', action='store_true')
    parser.add_argument('--tensorboard_parameters', action='store_true', default=True)

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False
        args.tensorboard_images = False
        args.tensorboard_parameters = False

    assert os.path.exists(args.dataset)
    args.milestones = json.loads(args.milestones)

    return args


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def convert_to_unit_vector(angles):
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    z = -torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi

args = parse_args()
def train(epoch, model, optimizer, criterion, train_loader, config, writer):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, poses, gazes) in enumerate(train_loader):
        global_step += 1

        if config['tensorboard_images'] and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        images = images.cuda()
        poses = poses.cuda()
        gazes = gazes.cuda()
        
        optimizer.zero_grad()

        outputs = model(images, poses)
        loss = criterion(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        if config['tensorboard']:
            writer.add_scalar('Train/RunningLoss.{}'.format(args.test_id), loss_meter.val, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'AngleError {:.2f} ({:.2f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            angle_error_meter.val,
                            angle_error_meter.avg,
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/AngleError{}'.format(args.test_id), angle_error_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, model, criterion, test_loader, config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    
    for step, (images, poses, gazes) in enumerate(test_loader):
        if config['tensorboard_images'] and epoch == 0 and step == 0:
            img = torchvision.utils.make_grid(
                        images, normalize=True, scale_each=True)      
            writer.add_image('Test/Image', img, epoch)
        
        #p = nn.ConstantPad2d((164, 164, 188, 188), 0)
        #images = p(images).resize_(64,1, 224, 224)

        images = images.cuda()
        poses = poses.cuda()
        gazes = gazes.cuda()
        #poses_test = poses[:, poses.shape[1]-1].cuda()
        poses_test = poses.flatten()
#         print('poses shape: ', poses_test.size())
#        print(poses[0])
        
#        print( 'im shape: ' , images.shape)
#         print(images[0])

        with torch.no_grad():
            outputs = model(images, poses)
            loss = criterion(outputs, gazes)

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

    logger.info('Epoch {} Loss {:.4f} AngleError {:.2f}'.format(
        epoch, loss_meter.avg, angle_error_meter.avg))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Test/AngleError{}'.format(args.test_id), angle_error_meter.avg, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    if config['tensorboard_parameters']:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    return angle_error_meter.avg


def main():
#    args = parse_args()
    logger.info(json.dumps(vars(args), indent=2))

    # TensorBoard SummaryWriter
    writer = SummaryWriter(filename_suffix='Ensemble_RLROP_{}'.format(args.test_id)) if args.tensorboard else None

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # data loaders
    train_loader, test_loader = get_loader(
        args.dataset, args.test_id, args.batch_size, args.num_workers, True)

    module = importlib.import_module('models.{}'.format(args.arch))
    vgg_att = module.VGG_ATT()
    model = module.Model(vgg_att)
    #modelB = module.ModelB()
#    model = module.Ensemble(modelA, modelB)
    model.cuda()
    
#    writer.add_graph(model, images)

#    for param in model.parameters:
#        param.require_grad = True

#    model.parameters.require_grad = True
#    for param in model.first_convlayer, model.fc1, model.fc2:
#        param.require_grad = True

    criterion = nn.MSELoss(size_average=True)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay)

    config = {
        'tensorboard': args.tensorboard,
        'tensorboard_images': args.tensorboard_images,
        'tensorboard_parameters': args.tensorboard_parameters,
    }

    # run test before start training
    test(0, model, criterion, test_loader, config, writer)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, criterion, train_loader, config, writer)
        angle_error = test(epoch, model, criterion, test_loader, config,
                           writer)
        #scheduler.step(angle_error)
        scheduler.step()

        state = OrderedDict([
            ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('angle_error', angle_error),
        ]) 
        model_path = os.path.join(outdir, 'model_state.pth')
        #torch.save(state, model_path)

    if args.tensorboard:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)


if __name__ == '__main__':
    main()

