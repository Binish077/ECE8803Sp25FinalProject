from __future__ import print_function
import pandas as pd
import math
import numpy as np
import torch
import torch.optim as optim
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from model import ResNet, EnhancedResnet50, MMT, MMCT
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from datasets import OLIVES, RECOVERY
import torch.nn as nn

def set_model(opt):
    """
    Function to set and initialize the model (set by user) + loss function (set to BCEWithLogitsLoss)
    """
    device = opt.device 

    # model selection based on user input
    if opt.model == "eresnet50":
        model = EnhancedResnet50(num_biomarkers=opt.ncls, pretrained=True)
    elif opt.model == "efficientnet-t":
        model = MMT(num_biomarkers=opt.ncls, texture_dim=12, use_texture=True)
    elif opt.model == "efficientnet-t-c":
        model = MMCT(num_biomarkers=opt.ncls, clinical_dim=4, texture_dim=12, use_texture=True)
    else:   
        model = ResNet(name=opt.model,num_classes = opt.ncls)
    
    # BCE loss for multi-label classification
    criterion = torch.nn.BCEWithLogitsLoss()

    # set the device (CUDA or CPU)
    model = model.to(device)
    criterion = criterion.to(device)

    return model, criterion


def set_loader(opt):
    """
    ONLY FOR TESTING WITH RECOVERY DATASET
    Function to set up data loaders for test and train
    """

    # construct data loader
    if opt.dataset == 'OLIVES' or opt.dataset == 'RECOVERY':
        # provided
        mean = (.1706)
        std = (.2112)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)

    # setting up random data transformations to enhance generalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    if opt.dataset =='OLIVES':
        csv_path_train = opt.train_csv_path
        csv_path_test = opt.test_csv_path
        data_path_train = opt.train_image_path
        data_path_test = opt.test_image_path
        train_dataset = OLIVES(csv_path_train,data_path_train,transforms = train_transform)
        test_dataset = RECOVERY(csv_path_test,data_path_test,transforms = val_transform)
    else:
        raise ValueError(opt.dataset)

    # shuffle the training dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    # create the test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,drop_last=False)

    return train_loader, test_loader

def set_loader_val(opt):
    """
    TO USE WITH TRAIN/TEST DATASET
    Function to set up data loaders for test and train
    """
    # Load the dataset
    if opt.dataset == 'OLIVES':
        mean = (.1706)
        std = (.2112)
        normalize = transforms.Normalize(mean=mean, std=std)

        csv_path = opt.train_csv_path
        data_path = opt.train_image_path
        df = pd.read_csv(csv_path)

        # randomly transform training dataset for generalization
        train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])

        # split into train and test sets (80, 20)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # dataloader, both test and train come from OLIVES
        train_loader = torch.utils.data.DataLoader(OLIVES(train_df, data_path, transforms=train_transform), 
                                batch_size=opt.batch_size, shuffle=True,
                                num_workers=opt.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(OLIVES(test_df, data_path, transforms=val_transform), 
                                batch_size=1, shuffle=False,
                                num_workers=0, pin_memory=True,drop_last=False)

        return train_loader, test_loader
    else:
        raise ValueError('Dataset not supported: {}'.format(opt.dataset))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):

    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)


    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state