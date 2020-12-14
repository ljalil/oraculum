import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from cmapssdataset import CMAPSSDataset

#%%
#torch.manual_seed(527)
#torch.manual_seed(1002)
#np.random.seed(589)


class CMAPSSBinaryClassifier(nn.Module):
    def __init__(self):
        super(CMAPSSBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(26, 16)
        self.fc2 = nn.Linear(16, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

def init_weights(mod):
    if type(mod) == nn.Linear:
        torch.nn.init.xavier_uniform_(mod.weight, gain=1.0)
        mod.bias.data.fill_(0.)

#%% Training process

data_path = "/home/abdeljalil/Workspace/Datasets/CMAPSS/"
data_FD01 = CMAPSSDataset(data_path, fd_number=1)
model_FD01 = CMAPSSBinaryClassifier()

#model_FD01.apply(init_weights)

loader_train, loader_test = data_FD01.construct_binary_classification_data(good_faulty_threshould=30, batch_size=64)
epochs = 10
optimizer = optim.Adam(model_FD01.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)
#optimizer = optim.SGD(model_FD01.parameters(), lr=0.001, momentum=0.9)

model_FD01.train()
criterion = nn.BCEWithLogitsLoss()

for epoch in range(epochs+1):
    correct = 0
    for batch_id, (data, target) in enumerate(loader_train):
        optimizer.zero_grad()
        output = model_FD01(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        output = torch.round(torch.sigmoid(output))
        correct += output.eq(target).sum().item()

    print('Train epoch: {}\t Accuracy ({:.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * correct/len(loader_train.dataset), loss.item()))

#%%
def train_cmapss_binary_classifier(model, optimizer, dataloader, epoch, logging_interval):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    correct = 0
    for batch_id, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        output = torch.round(torch.sigmoid(output))
        correct += output.eq(target).sum().item()

    print('Train epoch: {} ({:.0f}%)\t\tLoss: {:.6f}'.format(epoch, 100. * correct/len(dataloader.dataset), loss.item()))

def test_cmapss_binary_classifier(model, dataloader, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            test_loss += criterion(output, target).item()
            output = torch.round(torch.sigmoid(output))
            correct += output.eq(target).sum().item()

        print('Test Set: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(dataloader.dataset), 100.* correct/len(dataloader.dataset)))

def run_cmapss_binary_classifier():
    data_path = "/home/abdeljalil/Workspace/Datasets/CMAPSS/"
    data_FD01 = CMAPSSDataset(data_path, fd_number=1)
    model_FD01 = CMAPSSBinaryClassifier()

#    model_FD01.apply(init_weights)

    loader_train, loader_test = data_FD01.construct_binary_classification_data(good_faulty_threshould=30, batch_size=64)
    epochs = 10
    logging_interval = 64
    optimizer = optim.Adam(model_FD01.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)

    for epoch in range(epochs+1):
        train_cmapss_binary_classifier(model_FD01, optimizer, loader_train, epoch=epoch, logging_interval=logging_interval)
    test_cmapss_binary_classifier(model_FD01, loader_test, epoch)

run_cmapss_binary_classifier()


