import os
import sys
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms

import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def load_data(args):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if args.phase == "train":
        train_datasets = datasets.ImageFolder(os.path.join(args.data_dir, "train"),
                                            transform_train)
        trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)

    test_datasets = datasets.ImageFolder(os.path.join(args.data_dir, "test"),
                                          transform_test)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
        
    if args.phase == "train":
        return trainloader, testloader
    else:
        class_names = test_datasets.classes
        return testloader, class_names
