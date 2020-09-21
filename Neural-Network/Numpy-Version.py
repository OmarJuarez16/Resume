"""
Date: February/01/2020
Code - Feedforward Neural Network - Numpy Version
author: OmarJuarez16
@email: omarandre97@gmail.com
  
  Objective: This code represents a Neural Network with back propagation with 3 inputs (RGB), a hidden layer with 3 neurons and a
             output. The data is read from a .csv file with Pandas library.

"""

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    def __init__(self):
        # Parameters
        self.inputSize = _  # Write the number of inputs
        self.outputSize = _  # Write the number of outputs
        self.hiddenSize = _  # Write the number of neurons in hidden layer

        # Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (_, _) weight matrix
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (_, _) weight matrix

        # Learning rate
        self.lr = _ # Write the desired learning rate 

    def feedForward(self, x):  # Forward propagation for the network
        self.z = np.dot(x, self.W1)  # Dot product of input 'x' and first set of weights
        self.z2 = self.sigmoid(self.z)  # Activation function
        self.z3 = np.dot(self.z2, self.W2)  # Dot product of hidden layer (z2) and second set of weights (_x_)
        output = self.sigmoid(self.z3)  # Activation function
        return output

    def cost_function(self, prediction, target):  # Calculates cost function
        return -np.mean(np.sum(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)))

    def sigmoid(self, s, deriv=False):  # Sigmoid function (in Backward, Sigmoid differential is used)
        if deriv == True:
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def Backward(self, x, y, output):  # Backward propagation through the network
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        self.z2_error = self.output_delta.dot(self.W2.T)  # z2 error: how much hidden layer weights contribute to output
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)  # Applying derivative of sigmoid to z2 error

        self.W1 += x.T.dot(self.z2_delta)*self.lr  # Adjusting first set (Input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta)*self.lr  # Adjusting second set (hidden -> output) weights

    def train(self, x, y):  # Function to train the Neural Network
        output = self.feedForward(x)
        self.Backward(x, y, output)
        return self.W1, self.W2, output


def lecture_csv(filename='Datos.csv'):  # Read data from .csv
    r = []
    g = []
    b = []
    category = []
    global_data = pd.read_csv(filename, sep=',')  # Read data from .csv using Pandas library.
    global_data = np.asarray(global_data)
    size_array = np.shape(global_data)
    n_data = size_array[0]
    features = size_array[1] - 1

    for rows in range(n_data):
        b = np.append(b, global_data[rows, 0])
        g = np.append(g, global_data[rows, 1])
        r = np.append(r, global_data[rows, 2])
        category = np.append(category, global_data[rows, 3])

    return n_data, features, r, g, b, category


def feature_scaling(feature, mean, std):  # Feature scaling using Z-score normalization.
    return (feature - mean) / std


def main():
    # Define input data
    file_name = '..' # .csv file
    n_data, features, r, g, b, training_outputs = lecture_csv(filename=file_name)
    x = np.stack([b, g, r], axis=1)
    y = np.reshape(training_outputs, (np.shape(training_outputs)[0], 1))

    # Feature scaling
    mean = []
    std = []
    for i in range(3):
        mean = np.append(mean, np.mean(x[:, i]))
        std = np.append(std, np.std(x[:, i]))
        x[:, i] = feature_scaling(x[:, i], mean[i], std[i])

    nn = NeuralNetwork()  # Calling Neural Network class.
    cf = []
    stop_error = _  # Write the limit where it will stop training
    error = 1
    for i in range(_):  # Write the number of iterations for the NN
        w1, w2, output = nn.train(x, y) 
        cf = np.append(cf, nn.cost_function(output, y))
        if np.shape(cf)[0] > 1:
            error = np.abs(cf[-1] - cf[-2])
            counts = i
        if error <= stop_error:
            break
        print(cf[-1])
        print('This is the actual error: ', error)

    print('Loss: ', np.mean(np.square(y - nn.feedForward(x))))
    print('W1:', w1, 'W2: ', w2)

    # Plotting and displaying desired data.
    plt.plot(range(counts + 1), cf, 'k')  # Plot the error
    plt.xlabel('iterations')
    plt.ylabel('Cost')
    plt.grid()
    plt.show()
    x = cv.waitKey(0)

    if x & 0xFF == ord('q'):
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
