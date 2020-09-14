"""
Date: August/12/2020
Code - RGB Color Space
author: OmarJuarez16
@email: omarandre97@gmail.com
  
  Objective: This code analyzes the behavior of the RGB space against adversarial attacks. 
  
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
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torchvision
import os
from torch.utils.data import TensorDataset, DataLoader
from barbar import Bar


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

      x_adv = pgd_linf(model, x, y, eps, mode="Test")
      outputs = model(x.float() + x_adv.float())
      
      _, predicted = torch.max(outputs, 1)
      n_samples += labels.size(0)

      n_correct += (predicted.cpu() == labels).sum().item()
     
      for i in range(len(labels)):
        
        label = labels[i]
        pred = predicted[i]
      acc = n_correct / n_samples
      print('This is the accuracy: ', acc)
    
    return acc


def train_model(mode, dataset, dataloader, model, criterion, optimizer, trn_loss = [], trn_accuracy = [], tst_loss = [], tst_accuracy = []):
    
    if mode == 'train':
      model.train()
    else:  
      model.eval()
    
    cost = correct = 0

    for index, (feature, label) in enumerate(Bar(dataloader)):
      x, y = feature.to(device), label.to(device) 
      output = model(x.float())  
      y = torch.flatten(y).type(torch.LongTensor).to(device)

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

    
def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    total_loss, total_err = 0.,0.
    for X,y in Bar(loader):
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X.float()+delta.float())
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)    
    

def main():

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)  # Training dataset
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)  # Test dataset
    
    batch_size = 64
    
    # Dataloaders
    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)  # Batches and shuffling trainset
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)  # Batches and shuffling testset

    # Classes of CIFAR-10 
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #---------------------------------------------------------------------------
    # Normal training 
    
    Resnet_model = ResNet(BasicBlock, [2, 2, 2, 2], 3)
    Resnet_model.to(device)
    optimizer = optim.SGD(Resnet_model.parameters(), lr=0.1 , momentum = 0.9, weight_decay=1e-4, nesterov = True)
    loss_fn = nn.CrossEntropyLoss().to(device)

    global mother_path
    mother_path = ''  # Here goes the directory where you have all the files related to the training and testing. 
    
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    if os.path.isfile(mother_path + "Model_PGD0.pth"):
      print('Extracting pre-trained base model...')
      Resnet_model.load_state_dict(torch.load(mother_path + "Model_PGD0.pth"))  # Read the pre-trained model
      train_loss = np.genfromtxt(mother_path + "train_loss_PGD0.csv", delimiter=",").tolist()
      train_accuracy = np.genfromtxt(mother_path + "train_accuracy_PGD0.csv", delimiter=",").tolist()
      test_loss = np.genfromtxt(mother_path + "test_loss_PGD0.csv", delimiter=",").tolist()
      test_accuracy = np.genfromtxt(mother_path + "test_accuracy_PGD0.csv", delimiter=",").tolist()

    while len(train_loss) < 15:
        print('Generating base model...')
        train_loss, train_accuracy = train_model('train', train, trainset, Resnet_model, loss_fn, optimizer, trn_loss = train_loss, trn_accuracy = train_accuracy)
        with torch.no_grad():
          test_loss, test_accuracy = train_model('test', test, testset, Resnet_model, loss_fn, optimizer, trn_loss = train_loss, trn_accuracy = train_accuracy, tst_loss = test_loss, tst_accuracy = test_accuracy)
          torch.save(Resnet_model.state_dict(), mother_path + "Model_PGD0.pth")  # Save the model in directory
          np.savetxt(mother_path + "train_loss_PGD0.csv", train_loss, delimiter=",")
          np.savetxt(mother_path + "train_accuracy_PGD0.csv", train_accuracy, delimiter=",")
          np.savetxt(mother_path + "test_loss_PGD0.csv", test_loss, delimiter=",")
          np.savetxt(mother_path + "test_accuracy_PGD0.csv", test_accuracy, delimiter=",")
          print('Training loss: ', train_loss[-1], ', and this is the accuracy:', train_accuracy[-1])
          print('Test loss: ', test_loss[-1], ', and this is the accuracy:', test_accuracy[-1])

    ##---------------------------------------------------------------------------
    # Adversarial training - PGD-7
    Resnet_model_PGD7 = ResNet(BasicBlock, [2, 2, 2, 2])
    Resnet_model_PGD7.to(device)
    optimizer = optim.SGD(Resnet_model_PGD7.parameters(), lr=0.1 , momentum = 0.9, weight_decay=1e-4, nesterov = True) 
    loss_fn = nn.CrossEntropyLoss().to(device)

    train_loss_PGD7 = []
    train_accuracy_PGD7 = []
    test_loss_PGD7 = []
    test_accuracy_PGD7 = []

    if os.path.isfile(mother_path + "Model_PGD7.pth"):
      print('Extracting pre-trained base model...')
      Resnet_model_PGD7.load_state_dict(torch.load(mother_path + "Model_PGD7.pth"))  # Read the pre-trained model
      train_loss_PGD7 = np.genfromtxt(mother_path + "train_loss_PGD7.csv", delimiter=",").tolist()
      train_accuracy_PGD7 = np.genfromtxt(mother_path + "train_accuracy_PGD7.csv", delimiter=",").tolist()
      test_loss_PGD7 = np.genfromtxt(mother_path + "test_loss_PGD7.csv", delimiter=",").tolist()
      test_accuracy_PGD7 = np.genfromtxt(mother_path + "test_accuracy_PGD7.csv", delimiter=",").tolist()

    while len(test_loss_PGD7) < 20: 
      train_err, train_loss = epoch_adversarial(trainset, Resnet_model_PGD7, pgd_linf, optimizer)
      adv_err, adv_loss = epoch_adversarial(testset, Resnet_model_PGD7, pgd_linf)
      if len(test_accuracy_PGD7) > 1: 
        if adv_err > (1 - test_accuracy_PGD7[-1]) :
          for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 10 
      print(*("{:.6f}".format(i) for i in (train_err, adv_err)), sep="\t")
      train_loss_PGD7.append(train_loss)
      train_accuracy_PGD7.append(1-train_err)
      test_loss_PGD7.append(adv_loss)
      test_accuracy_PGD7.append(1-adv_err)
      torch.save(Resnet_model_PGD7.state_dict(), mother_path + "Model_PGD7.pth")  # Save the model in directory
      np.savetxt(mother_path + "train_loss_PGD7.csv", train_loss_PGD1, delimiter=",")
      np.savetxt(mother_path + "train_accuracy_PGD7.csv", train_accuracy_PGD7, delimiter=",")
      np.savetxt(mother_path + "test_loss_PGD7.csv", test_loss_PGD7, delimiter=",")
      np.savetxt(mother_path + "test_accuracy_PGD7.csv", test_accuracy_PGD7, delimiter=",")
    
    ##---------------------------------------------------------------------------
    ## In this section, the models are evaluated

    Resnet_model_PGD7.eval()      
    pgd_attack_range = [0, 5/255, 10/255, 15/255, 20/255, 25/255, 30/255]
    acurracy = []

    for eps in pgd_attack_range: 
      acurracy.append(eval_against_adv(testset, Resnet_model_PGD7, eps=eps))
    
    plt.figure(figsize=(15,10))
    plt.title('$L_{\inf}$-bounded adversary')
    plt.plot(pgd_attack_range, acurracy, label='CIELAB PGD-7')
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
