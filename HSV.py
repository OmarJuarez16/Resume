"""
Date: August/12/2020
Code - HSV Color Space
author: OmarJuarez16
@email: omarandre97@gmail.com
  
  Objective: This code analyzes the behavior of the HSV space against adversarial attacks. 
  
  References: 
  - Towards Deep Learning Models Resistant To Adversarial Attack — Madry et al 2017
  
"""


# Importing libraries
from Models import *
from Attack import * 
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kornia import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from collections import OrderedDict
import os
from torch.utils.data import TensorDataset, DataLoader
from barbar import Bar
from math import pi 


def eval_against_adv(testset, model, eps):
    total = 0
    acc = 0
    model.eval()
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
   
    for index, (feature, labels) in enumerate(Bar(testset)):
      x, y = feature.to(device), labels.to(device) 

      hsv = rgb_to_hsv(x)                                   # HSV transform
      hsv[:, 0, :, :] = hsv[:, 0, :, :] / (pi * 2)          # Normalize Hue channel
      delta_hsv = pgd_linf(model, hsv, y, eps, mode="Test") # Obtain HSV-attack
      hsv[:, 0, :, :] = hsv[:, 0, :, :] * (pi * 2)          # Return Hue channel range 
      new_rgb = hsv_to_rgb(hsv + delta_hsv)                 # RGB attacked image
      delta_rgb = new_rgb - x                               # Resulting delta for RGB
      final_delta = delta_rgb.clamp(-eps, eps)              # Clamping the delta
      final_hsv = rgb_to_hsv(x + final_delta)               # Resulting HSV image
      final_hsv[:, 0, :, :] = final_hsv[:, 0, :, :] / (pi * 2)

      outputs = model(final_hsv.float())
      
      _, predicted = torch.max(outputs, 1)
      n_samples += labels.size(0)

      n_correct += (predicted.cpu() == labels).sum().item()
     
      for i in range(len(labels)):
        
        label = labels[i]
        pred = predicted[i]
      acc = n_correct / n_samples
      print('This is the accuracy: ', acc)

    return acc


def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    total_loss, total_err = 0.,0.
    i = 0
    for x,y in Bar(loader):
        x, y = x.to(device), y.to(device)
        hsv = rgb_to_hsv(x)                           # HSV transform
        hsv[:, 0, :, :] = hsv[:, 0, :, :] / (pi * 2)  # Normalize Hue channel
        delta_hsv = attack(model, hsv, y, **kwargs)   # Obtain HSV-attack
        hsv[:, 0, :, :] = hsv[:, 0, :, :] * (pi * 2)  # Return Hue channel range 
        new_rgb = hsv_to_rgb(hsv + delta_hsv)         # RGB attacked image
        delta_rgb = new_rgb - x                       # Resulting delta for RGB
        final_delta = delta_rgb.clamp(-8/255, 8/255)  # Clamping the delta
        final_hsv = rgb_to_hsv(x + final_delta)       # Resulting HSV image
        final_hsv[:, 0, :, :] = final_hsv[:, 0, :, :] / (pi * 2)  # Normalizing

        yp = model(final_hsv.float())
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
      
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * x.shape[0]
        i += 1
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def train_model(mode, dataset, dataloader, model, criterion, optimizer, trn_loss = [], trn_accuracy = [], tst_loss = [], tst_accuracy = []):
    
    if mode == 'train':
      model.train()
    else:  
      model.eval()
    
    cost = correct = 0

    for index, (feature, label) in enumerate(Bar(dataloader)):
      x, y = feature.to(device), label.to(device) 
      x = rgb_to_hsv(x)
      x[:, 0, :, :] = x[:, 0, :, :] / (pi * 2)
      output = model(x.float())  
      y = torch.flatten(y).type(torch.LongTensor).cuda()

      loss = criterion(output, y)  

      if mode == 'train': 
        loss.backward()  
        optimizer.step()
        optimizer.zero_grad() 

      cost += loss.item() * feature.shape[0]
      correct += (output.argmax(1) == label.cuda()).sum().item()
    
    cost = cost / len(dataset) 
    acc = correct / len(dataset)

    if mode == 'train':                     
      trn_loss.append(cost)
      trn_accuracy.append(acc)
      return trn_loss, trn_accuracy

    else: 
      if len(tst_accuracy) > 1: 
        if acc < tst_accuracy[-1]: 
          for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10  
          
      tst_loss.append(cost)
      tst_accuracy.append(acc)
      
      return tst_loss, tst_accuracy


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
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #---------------------------------------------------------------------------
    # Normal training 
    Dense_model = DenseNet(3)
    Dense_model.to(device)
    optimizer = optim.SGD(Dense_model.parameters(), lr=0.1 , momentum = 0.9, weight_decay=1e-4, nesterov = True)
    loss_fn = nn.CrossEntropyLoss().cuda()
    
    global mother_path
    mother_path = ''  # Here goes the directory where you have all the files related to the training and testing. 
    
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    if os.path.isfile(mother_path + "Model_PGD0.pth"):
      print('Extracting pre-trained base model...')
      Dense_model.load_state_dict(torch.load(mother_path + "Model_PGD0.pth"))  # Read the pre-trained model
      train_loss = np.genfromtxt(mother_path + "train_loss_PGD0.csv", delimiter=",").tolist()
      train_accuracy = np.genfromtxt(mother_path + "train_accuracy_PGD0.csv", delimiter=",").tolist()
      test_loss = np.genfromtxt(mother_path + "test_loss_PGD0.csv", delimiter=",").tolist()
      test_accuracy = np.genfromtxt(mother_path + "test_accuracy_PGD0.csv", delimiter=",").tolist()

    if len(train_loss) == 0: 
      print("Generating base model...")
    if 0 < len(train_loss) < 20:
      print("Still not done, continuing training the model...")
    while len(train_loss) < 20:
        train_loss, train_accuracy = train_model('train', train, trainset, Dense_model, loss_fn, optimizer, trn_loss = train_loss, trn_accuracy = train_accuracy)
        with torch.no_grad():
          test_loss, test_accuracy = train_model('test', test, testset, Dense_model, loss_fn, optimizer, trn_loss = train_loss, trn_accuracy = train_accuracy, tst_loss = test_loss, tst_accuracy = test_accuracy)
          torch.save(Dense_model.state_dict(), mother_path + "Model_PGD0.pth")  # Save the model in directory
          np.savetxt(mother_path + "train_loss_PGD0.csv", train_loss, delimiter=",")
          np.savetxt(mother_path + "train_accuracy_PGD0.csv", train_accuracy, delimiter=",")
          np.savetxt(mother_path + "test_loss_PGD0.csv", test_loss, delimiter=",")
          np.savetxt(mother_path + "test_accuracy_PGD0.csv", test_accuracy, delimiter=",")
          print('Training loss: ', train_loss[-1], ', and this is the accuracy:', train_accuracy[-1])
          print('Test loss: ', test_loss[-1], ', and this is the accuracy:', test_accuracy[-1])

    ##---------------------------------------------------------------------------
    # Adversarial training - PGD-7
    Dense_model_PGD7 = DenseNet(3)
    Dense_model_PGD7.to(device)
    optimizer = optim.SGD(Dense_model_PGD7.parameters(), lr=0.1 , momentum = 0.9, weight_decay=1e-4, nesterov = True)
    loss_fn = nn.CrossEntropyLoss().cuda()

    train_loss_PGD7 = []
    train_accuracy_PGD7 = []
    test_loss_PGD7 = []
    test_accuracy_PGD7 = []

    if os.path.isfile(mother_path + "Model_PGD7.pth"):
      print('Extracting pre-trained adversarial trained model...')
      Dense_model_PGD7.load_state_dict(torch.load(mother_path + "Model_PGD7.pth"))  # Read the pre-trained model
      train_loss_PGD7 = np.genfromtxt(mother_path + "train_loss_PGD7.csv", delimiter=",").tolist()
      train_accuracy_PGD7 = np.genfromtxt(mother_path + "train_accuracy_PGD7.csv", delimiter=",").tolist()
      test_loss_PGD7 = np.genfromtxt(mother_path + "test_loss_PGD7.csv", delimiter=",").tolist()
      test_accuracy_PGD7 = np.genfromtxt(mother_path + "test_accuracy_PGD7.csv", delimiter=",").tolist()

    if len(test_loss_PGD7) == 0: 
      print("Generating adversarial trained model...")
    if 0 <len(test_loss_PGD7) < 20:
      print("Still not done, continuing training the model...")
    while len(test_loss_PGD7) < 20: 
      train_err, train_loss = epoch_adversarial(trainset, Dense_model_PGD7, pgd_linf, optimizer)
      adv_err, adv_loss = epoch_adversarial(testset, Dense_model_PGD7, pgd_linf)
      if len(test_accuracy_PGD7) > 1: 
        if adv_err > (1 - test_accuracy_PGD7[-1]) :
          for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 10 
      print(*("{:.6f}".format(i) for i in (1-train_err, 1-adv_err)), sep="\t")
      train_loss_PGD7.append(train_loss)
      train_accuracy_PGD7.append(1-train_err)
      test_loss_PGD7.append(adv_loss)
      test_accuracy_PGD7.append(1-adv_err)
      torch.save(Dense_model_PGD7.state_dict(), mother_path + "Model_PGD7.pth")  # Save the model in directory
      np.savetxt(mother_path + "train_loss_PGD7.csv", train_loss_PGD7, delimiter=",")
      np.savetxt(mother_path + "train_accuracy_PGD7.csv", train_accuracy_PGD7, delimiter=",")
      np.savetxt(mother_path + "test_loss_PGD7.csv", test_loss_PGD7, delimiter=",")
      np.savetxt(mother_path + "test_accuracy_PGD7.csv", test_accuracy_PGD7, delimiter=",")
    
    ##---------------------------------------------------------------------------
    ## In this section, the model is evaluated
    
    Dense_model_PGD7.eval()      
    pgd_attack_range = [0/255, 5/255, 10/255, 15/255, 20/255, 25/255, 30/255]
    acurracy = []

    for eps in pgd_attack_range: 
      acurracy.append(eval_against_adv(testset, Dense_model_PGD7, eps=eps, alpha=2.5*eps/100, n_iter=100))
    
    plt.figure(figsize=(15,10))
    plt.title('$L_{\inf}$-bounded adversary')
    plt.plot(pgd_attack_range, acurracy, label='HSV PGD-7')
    plt.ylabel('Accuracy')
    plt.xlabel('Epsilon')
    plt.ylim((0,1))
    plt.legend()
    plt.grid()
    plt.savefig('epsilons.png', dpi=100)
    plt.show()
    print('This is the list of accuracies: ', acurracy)

    np.savetxt(mother_path + "Accuracy-against-PGD-7.csv", acurracy, delimiter=",")
      


if __name__ == "__main__":
    main()
    
