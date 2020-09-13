"""
Date: August/12/2020
Code - ResNet Model
author: OmarJuarez16
@email: omarandre97@gmail.com
  
  Objective: This code analyzes the behavior of the HSV space against adversarial attacks. The 
  
"""


# Importing libraries
from models import *
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
from skimage.io import imread, imshow
import cv2 as cv
from torch.utils.data import TensorDataset, DataLoader
from barbar import Bar


def main():
    
    transform = transforms.Compose([transforms.ToTensor()]) # No transformations needed for HSV space
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  # Training dataset
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  # Test dataset
    
    batch_size = 64
    
    # Dataloaders
    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)  
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)  

    # Classes of CIFAR-10 
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #---------------------------------------------------------------------------
    # Normal training 
    
    Resnet_model = ResNet(BasicBlock, [2, 2, 2, 2], 3)  # ResNet-18 model
    
