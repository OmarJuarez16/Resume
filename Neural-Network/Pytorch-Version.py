"""
Date: August/27/2020
PEF Code - Grape cluster segmentation code from PPD
author: OmarJuarez16
@email: omarandre97@gmail.com

    Purpose: The purpose of this code is to segment the grape clusters from a .png image by using a Feedforward NN
             model using Pytorch library.
"""

#  Importing libraries
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from barbar import Bar
import os
import cv2 as cv
import glob


class Feedforward(nn.Module):
    """
    Neural Network
        Model: Feedforward
        Architecture: 1 hidden layer with 3 neurons
        Activation:
          - Hidden layer: ReLu
          - Output layer: Sigmoid function
    """
    def __init__(self):
        super(Feedforward, self).__init__()
        # Layers of NN
        self.input_layer = nn.Linear(3, 5)  # Input - Hidden 1
        self.hidden_layer = nn.Linear(5, 5)  # Hidden 1 - Hidden 2
        self.output_layer = nn.Linear(5, 1)  # Hidden 2 - Output

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        return self.sigmoid(x)


def train_model(device, data_loader, model, criterion, optimizer, lst_loss):

    for index, (feature, label) in enumerate(Bar(data_loader)):
        x, y = feature.to(device), label.to(device)
        x = x.float()
        x.requires_grad = True
        output = model(x)
        loss = criterion(output.flatten(), y.flatten())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if len(lst_loss) > 0:
            if lst_loss[-1] < loss:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 10

    return loss


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("Batch_size", default=16, help="Introduce the batch size")
    parser.add_argument("Epochs", default=5, help="Introduce the number of epochs for the training")
    args = parser.parse_args()

    batch = int(args.Batch_size)
    epochs = int(args.Epochs)

    directory = ""  # Introduce the directory of the images
    positive = os.listdir(directory + "Positive/")
    negative = os.listdir(directory + "Negative/")
    features = []
    labels = []
    for image in positive:
        image = cv.imread(directory + "/Positive/" + image)
        image = np.reshape(image, (np.shape(image)[0] * np.shape(image)[1], 3))
        features.extend(image)
        labels.extend(np.ones((np.shape(image)[0])))
    for image in negative:
        image = cv.imread(directory + "/Negative/" + image)
        image = np.reshape(image, (np.shape(image)[0] * np.shape(image)[1], 3))
        features.extend(image)
        labels.extend(np.zeros((np.shape(image)[0])))

    train_data = []
    for i in range(len(features)):
        if features[i][0] != 0 and features[i][1] != 0 and features[i][2] != 0:
            train_data.append([features[i], labels[i]])

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NN = Feedforward()
    NN.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(NN.parameters(), lr = 0.1)

    train_loss = []

    if os.path.isfile("./.../Model.pth"):
        NN.load_state_dict(torch.load(".../Model.pth"))
        train_loss = np.genfromtxt(".../train-loss.csv", delimiter=",")

    if len(train_loss) < epochs:
        print("Generating model...")
    else:
        print("Model is ready!")
    while len(train_loss) < epochs:
        loss = train_model(device, train_loader, NN, criterion, optimizer, train_loss)
        print('Actual loss:', loss.detach().cpu())
        torch.save(NN.state_dict(), '.../Model.pth')  # Saves the model as Model.pth
        train_loss.append(loss)
        np.savetxt(".../train-loss.csv", train_loss, delimiter=",")  # Saves the loss list

    NN.eval()
    path = glob.glob(".../*.png")  # Path were images are found.

    for images in path:

        image = cv.imread(images)
        image2 = image.copy()
        output = NN(torch.from_numpy(image2).float()).detach().numpy()

        for i in range(np.shape(image2)[0]):
            for j in range(np.shape(image2)[1]):
                if output[i, j] < 0.5:
                    image2[i, j] = 0

        cv.namedWindow('Image', flags=cv.WINDOW_NORMAL)
        cv.imshow('Image', image2)

        x = cv.waitKey(0)

        if x & 0xFF == ord('q'):
            cv.destroyAllWindows()

        cv.imwrite("", image3)


if __name__ == "__main__":
    main()
