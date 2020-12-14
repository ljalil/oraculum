import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import importlib
import cmapssdataset

#%%

class CMAPSSFullyConnectedRUL(nn.Module):
    def __init__(self):
        super( CMAPSSFullyConnectedRUL, self).__init__()
        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        output = self.fc4(x)
        return output


data_path = "/home/abdeljalil/Workspace/Datasets/CMAPSS/"
data_FD01 = cmapssdataset.CMAPSSDataset(data_path, fd_number=1)
model_FD01 = CMAPSSFullyConnectedRUL()

loader_train, loader_test = data_FD01.construct_regression_data()

epochs = 100
optimizer = optim.Adam(model_FD01.parameters(), lr=0.01, betas=(0.9, 0.999))
#optimizer = optim.SGD(model_FD01.parameters(), lr=0.01, momentum=0.9)

model_FD01.train()
criterion = nn.MSELoss()

for epoch in range(epochs+1):
    for batch_id, (data, target) in enumerate(loader_train):
        optimizer.zero_grad()
        output = model_FD01(data)
        output = output.view_as(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Train epoch: {}\t \tLoss: {:.6f}'.format(epoch, loss.item()))

#%%
:
