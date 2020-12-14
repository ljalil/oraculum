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

path_data = '/home/abdeljalil/Workspace/casewestern/'

NB = ['Normal_0.mat', 'Normal_1.mat', 'Normal_2.mat', 'Normal_3.mat']

B07 = ['B007_0.mat', 'B007_1.mat', 'B007_2.mat', 'B007_3.mat']
B14 = ['B014_0.mat', 'B014_1.mat', 'B014_2.mat', 'B014_3.mat']
B21 = ['B021_0.mat', 'B021_1.mat', 'B021_2.mat', 'B021_3.mat']

IR07 = ['IR007_0.mat', 'IR007_1.mat', 'IR007_2.mat', 'IR007_3.mat']
IR14 = ['IR014_0.mat', 'IR014_1.mat', 'IR014_2.mat', 'IR014_3.mat']
IR21 = ['IR021_0.mat', 'IR021_1.mat', 'IR021_2.mat', 'IR021_3.mat']

OR07 = ['OR007@6_0.mat', 'OR007@6_1.mat', 'OR007@6_2.mat', 'OR007@6_3.mat']
OR14 = ['OR014@6_0.mat', 'OR014@6_1.mat', 'OR014@6_2.mat', 'OR014@6_3.mat']
OR21 = ['OR021@6_0.mat', 'OR021@6_1.mat', 'OR021@6_2.mat', 'OR021@6_3.mat']

full_data = [NB, B07, B14, B21, IR07, IR14, IR21, OR07, OR14, OR21]

#%%

data_x = np.zeros((0, 1, 64, 64))
data_y = np.zeros((0,))

for bearing_state in enumerate(full_data):
    for load in bearing_state[1]:
        data = loadmat(os.path.join(path_data, load))
        vibration = data[list(data.keys())[3]]
        number_of_samples = (vibration.shape[0]//(64*64))
        usable_length = number_of_samples*64*64
        vibration = vibration[:usable_length]
        vibration = vibration.reshape(-1, 1, 64, 64)
        data_x = np.concatenate((data_x, vibration), axis=0)
        labels = np.ones((1,)) * bearing_state[0]
        labels = np.repeat(labels, number_of_samples, axis=0)
        data_y = np.concatenate((data_y, labels), axis=0)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

tensor_train_x = torch.Tensor(train_x)
tensor_train_y = torch.LongTensor(train_y)
tensor_test_x = torch.Tensor(test_x)
tensor_test_y = torch.LongTensor(test_y)

batch_size = 64
loader_train = DataLoader(TensorDataset(tensor_train_x, tensor_train_y), batch_size=batch_size)
loader_test = DataLoader(TensorDataset(tensor_test_x, tensor_test_y), batch_size=batch_size)


#%%

class CaseWesternClassifier(nn.Module):
    def __init__(self):
        super(CaseWesternClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.fc1 = nn.Linear(5184, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = x.view(-1,5184 )
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
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id*len(data), len(dataloader.dataset), 100. * batch_id*len(dataloader), loss.item()))


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

clf = CaseWesternClassifier()
optimizer = optim.Adam(clf.parameters())
log_interval = 50
#%%

for epoch in range(1,21):
    train(clf, loader_train, optimizer, epoch, log_interval) 
    test(clf, loader_test, epoch)
