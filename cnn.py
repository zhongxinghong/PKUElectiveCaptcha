#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: cnn.py
# Created Date: Wednesday, January 8th 2020, 6:12:20 pm
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

import os
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils import u
from const import LOG_DIR, CNN_MODEL_FILE, LABELS_NUM
from dataset import ElectiveCaptchaDatasetFromPackage


CONFUSION_MATRIX_LOG_FILE = os.path.join(LOG_DIR, r"cnn.confusion_matrix.epoch_{}.csv")


class ElectiveCaptchaCNN(nn.Module):

    def __init__(self):
        super(ElectiveCaptchaCNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, LABELS_NUM) # 55
        
    def forward(self, x):
        x = self.conv1(x)       # batch*32*20*20
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # batch*32*10*10
        x = self.conv2(x)       # batch*64*8*8
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # batch*64*4*4
        x = self.conv3(x)       # batch*128*2*2
        x = self.bn3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1) # batch*512
        x = self.fc1(x)         # batch*128
        x = F.relu(x)
        x = self.fc2(x)         # batch*55
        x = F.log_softmax(x, dim=1)
        return x


def train(model, train_loader, optimizer, epoch):
    
    log_interval = int(len(train_loader) * 0.05)

    model.train()

    for ix, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if ix % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                ix * len(data), 
                len(train_loader.sampler),
                100.0 * ix / len(train_loader), 
                loss.item()
            ))


def validate(model, validation_loader, epoch):

    model.eval()
    validation_loss = 0
    correct = 0

    confusion_matrix = np.zeros((LABELS_NUM, LABELS_NUM), dtype=np.int)
    
    with torch.no_grad():
        for Xlist, ylist in validation_loader:
            output = model(Xlist)
            validation_loss += F.nll_loss(output, ylist).item() / len(validation_loader.sampler)
            ypred = output.argmax(dim=1, keepdim=True)
            correct += ypred.eq(ylist.view_as(ypred)).sum().item()
            for t, p in zip(ylist.view(-1), ypred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print('\nValidation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        validation_loss, 
        correct, 
        len(validation_loader.sampler),
        100.0 * correct / len(validation_loader.sampler)
    ))

    df = pd.DataFrame(
        data=confusion_matrix,
        index=validation_loader.dataset.labels,
        columns=validation_loader.dataset.labels,  
    )
    df.to_csv(CONFUSION_MATRIX_LOG_FILE.format(epoch))


def main():

    RANDOM_STATE = 42
    TRAIN_SIZE = 0.7
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 0.1
    LR_STEP_SIZE = 1
    LR_STEP_GAMMA = 0.15

    dataset = ElectiveCaptchaDatasetFromPackage()

    indices = np.arange(len(dataset))
    
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(indices)
    
    sep = int(len(dataset) * TRAIN_SIZE)

    train_indices, validation_indices = indices[:sep], indices[sep:]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

    model = ElectiveCaptchaCNN()
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_STEP_GAMMA)

    for epoch in range(1, EPOCHS+1):
        train(model, train_loader, optimizer, epoch)
        validate(model, validation_loader, epoch)
        scheduler.step()

    joblib.dump(model.state_dict(), CNN_MODEL_FILE, compress=9)


if __name__ == '__main__':
    main()

