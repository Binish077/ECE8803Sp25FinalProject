import torch
import argparse
from utils import AverageMeter,save_model
import sys
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from config import parse_option
import os
from utils import set_loader, set_loader_val, set_model, set_optimizer, adjust_learning_rate
import torch.serialization
import pandas as pd
from pathlib import Path


def train_supervised(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training for multi-modal input"""

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()

    for idx, (image, clinical, texture, labels) in enumerate(train_loader):
        data_time.update(time.time() - end) # keeping track of time

        # prepare inputs to the model
        image = image.to(device)
        clinical = clinical.to(device)
        texture = texture.to(device)
        labels = labels.float().to(device)
        bsz = labels.shape[0]

        # compute loss
        output = model(image, clinical, texture)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'.format(
                epoch, idx + 1, len(train_loader)))
            sys.stdout.flush()

    return losses.avg


def submission_generate(val_loader, model, opt):
    """FOR RECOVERY DATASET TESTING ONLY"""
    model.eval()

    device = opt.device
    out_list = []
    with torch.no_grad():
        for idx, (image) in (enumerate(val_loader)):

            images = image.float().to(device)

            # forward
            output = model(images)
            output = torch.round(torch.sigmoid(output))
            out_list.append(output.squeeze().detach().cpu().numpy())


    out_submisison = np.array(out_list)

def evaluate_model(test_loader, model, opt):
    """FOR TEST/TRAIN SPLIT ONLY"""
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for images, clinical, texture, targets in test_loader:

            # prepare data
            images = images.to(opt.device)
            clinical = clinical.to(opt.device)
            texture = texture.to(opt.device)
            targets = targets.to(opt.device)

            # test the model
            outputs = model(images, clinical, texture)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            # save predictions and labels
            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    # save predictions and labels as .npy files
    predictions_file = './save/test_predictions.npy'
    labels_file = './save/test_labels.npy'
    np.save(predictions_file, predictions)
    np.save(labels_file, labels)
    print(f"Predictions saved to {predictions_file}")
    print(f"Labels saved to {labels_file}")

    # metrics
    accuracy = accuracy_score(labels, predictions) # accuracy score
    f1 = f1_score(labels, predictions, average='macro') # f1 macro
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")

    return predictions_file, labels_file

def main():
    """
    Train and test model
    """
    opt = parse_option()

    # if testing on RECOVERY:
    #train_loader,test_loader = set_loader(opt)

    # if using a test/train split:
    train_loader, test_loader = set_loader_val(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        train_supervised(train_loader, model, criterion, optimizer, epoch, opt)

    # save the model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # test the model (for test/train split)
    evaluate_model(test_loader, model, opt)

    # if testing on RECOVERY:
    # submission_generate(test_loader, model, opt)

if __name__ == '__main__':
    main()