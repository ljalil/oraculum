"""
Paper: A New Convolutional Neural Network Based Data-Driven Fault Diagnosis Method
Authors: Long Wen, Xinyu Li, Liang Gao and Yuyan Zhang
Journal: IEEE Transactions on Industrial Electronics
Year: 2018
DOI: 10.1109/TIE.2017.2774777
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def prepare_data(data_path, batch_size, test_size):

    # Each list contains data files for a specific fault type and size, in addition
    # to the normal baseline data

    # Normal baseline data
    NB = ['Normal_0.mat', 'Normal_1.mat', 'Normal_2.mat', 'Normal_3.mat']

    # Ball fault with 0.007", 0.014" and 0.021" fault diameters
    B07 = ['B007_0.mat', 'B007_1.mat', 'B007_2.mat', 'B007_3.mat']
    B14 = ['B014_0.mat', 'B014_1.mat', 'B014_2.mat', 'B014_3.mat']
    B21 = ['B021_0.mat', 'B021_1.mat', 'B021_2.mat', 'B021_3.mat']

    # Inner race fault with 0.007", 0.014" and 0.021" fault diameters
    IR07 = ['IR007_0.mat', 'IR007_1.mat', 'IR007_2.mat', 'IR007_3.mat']
    IR14 = ['IR014_0.mat', 'IR014_1.mat', 'IR014_2.mat', 'IR014_3.mat']
    IR21 = ['IR021_0.mat', 'IR021_1.mat', 'IR021_2.mat', 'IR021_3.mat']

    # Outer race fault with 0.007", 0.014" and 0.021" fault diameters and load zone
    # centered at 6:00
    OR07 = ['OR007@6_0.mat', 'OR007@6_1.mat', 'OR007@6_2.mat', 'OR007@6_3.mat']
    OR14 = ['OR014@6_0.mat', 'OR014@6_1.mat', 'OR014@6_2.mat', 'OR014@6_3.mat']
    OR21 = ['OR021@6_0.mat', 'OR021@6_1.mat', 'OR021@6_2.mat', 'OR021@6_3.mat']

    # Each element in full_data contains all recordings corresponding to fault
    # type  and diameter (plus normal baseline) and will be used to iterate over
    # and process it
    full_data = [NB, B07, B14, B21, IR07, IR14, IR21, OR07, OR14, OR21]

    # data_x and data_y will contain the whole constructed dataset
    # data_x will contain the 64*64 images
    # data_y will contain the 10 classes labels
    data_x = np.zeros((0, 1, 64, 64))
    data_y = np.zeros((0,))
    for bearing_state in enumerate(full_data):
        for load in bearing_state[1]:
            data = loadmat(os.path.join(data_path, load))
            vibration = data[list(data.keys())[3]]

            # Calculate the number of samples that can be extracted from each
            # recording in order to construct 64*64 images
            number_of_samples = (vibration.shape[0]//(64*64))

            # Calculate the usable length of each recording, in order to discard
            # the last part that isn't enough to make a 64*64 image
            usable_length = number_of_samples*64*64
            vibration = vibration[:usable_length]

            # The rest of the data is reshaped into a set of 64*64 images
            vibration = vibration.reshape(-1, 1, 64, 64)

            # The extracted and reshaped data is appended to variable data_x
            data_x = np.concatenate((data_x, vibration), axis=0)

            # Labels are numbers from 0 to 9 representing the 10 bearing states
            # They are generated for each working condition and appended to data_y
            labels = np.ones((1,)) * bearing_state[0]
            labels = np.repeat(labels, number_of_samples, axis=0)
            data_y = np.concatenate((data_y, labels), axis=0)

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, shuffle=True)

    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.LongTensor(train_y)
    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.LongTensor(test_y)

    batch_size = batch_size
    loader_train = DataLoader(TensorDataset(tensor_train_x, tensor_train_y), batch_size=batch_size)
    loader_test = DataLoader(TensorDataset(tensor_test_x, tensor_test_y), batch_size=batch_size)

    return loader_train, loader_test

class CaseWesternClassifier(nn.Module):
    def __init__(self):
        super(CaseWesternClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output

def train(model, dataloader, optimizer, epoch, log_interval):
    model.train()
    criterion = nn.NLLLoss()

    for batch_id, (data, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()

        if batch_id % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id*len(data), len(dataloader.dataset), 100. * batch_id/len(dataloader), loss.item()))


def test(model, dataloader, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, targets in dataloader:
            output = model(data)
            test_loss += F.nll_loss(output, targets, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(targets).sum().item()

            test_loss /= len(dataloader.dataset)
        print('Test Set: Average Loss {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(dataloader.dataset), 100.* correct/len(dataloader.dataset)))

if __name__ == "__main__":
    data_path = input("Enter data path: ")
    epochs = input("Enter training epochs (default=30): ")
    batch_size = input("Enter batch size (default=64): ")
    test_size = input("Enter test size (default=0.2): ")
    log_interval = input("Enter log interval (default=1): ")

    if not epochs: epochs=30
    else: epochs=int(epochs)

    if not batch_size: batch_size=64
    else: batch_size=int(batch_size)

    if not test_size: test_size=0.2
    else: test_size=float(test_size)

    if not log_interval: log_interval=1
    else: log_interval=int(log_interval)

    clf = CaseWesternClassifier()
    optimizer = optim.Adam(clf.parameters())

    loader_train, loader_test = prepare_data(data_path, batch_size, test_size)

    for epoch in range(1,epochs+1):
        train(clf, loader_train, optimizer, epoch, log_interval)
        test(clf, loader_test, epoch)
