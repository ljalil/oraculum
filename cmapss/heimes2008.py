"""
Paper: Recurrent Neural Networks for Remaining Useful Life Estimation
Authors: Felix O. Heimes
Journal: IEEE
Year: 2008
DOI: 10.1109/PHM.2008.4711422
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from cmapssdataset import CMAPSSDataset

class CMAPSSBinaryClassifier(nn.Module):
    """
    This class performs binary classification using a simple fully-connected neural network
    (i.e. healthy/faulty engines) by assuming first and last x samples belonging to each
    engine correspond to healthy and faulty states respectively.
    """
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

def run_cmapss_binary_classifier(data_path, fd_number, good_faulty_threshold, lr, batch_size, epochs, logging_interval):
    data_FD0x = CMAPSSDataset(data_path, fd_number)
    cmapss_classifier = CMAPSSBinaryClassifier()

    loader_train, loader_test = data_FD0x.construct_binary_classification_data(good_faulty_threshold, batch_size)
    optimizer = optim.Adam(cmapss_classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)

    for epoch in range(epochs+1):
        cmapss_classifier.train()
        criterion = nn.BCEWithLogitsLoss()
        correct = 0
        for batch_id, (data, target) in enumerate(loader_train):
            optimizer.zero_grad()
            output = cmapss_classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            output = torch.round(torch.sigmoid(output))
            correct += output.eq(target).sum().item()
        print('Train epoch: {} ({:.0f}%)\t\tLoss: {:.6f}'.format(epoch, 100. * correct/len(loader_train.dataset), loss.item()))

    # Model testing
    cmapss_classifier.eval()
    test_loss = 0.0
    correct = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data, target in loader_test:
            output = cmapss_classifier(data)
            test_loss += criterion(output, target).item()
            output = torch.round(torch.sigmoid(output))
            correct += output.eq(target).sum().item()

        print('Test Set: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(loader_test.dataset), 100.* correct/len(loader_test.dataset)))

class CMAPSSFCNNRegressor(nn.Module):
    def __init__(self):
        super (CMAPSSFCNNRegressor, self).__init__()
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

def run_cmapss_fcnn_regressor(data_path, fd_number, rul_type, lr, batch_size, epochs, logging_interval):
    data_FD0x = CMAPSSDataset(data_path, fd_number)
    cmapss_regressor = CMAPSSFCNNRegressor()

    loader_train, loader_test = data_FD0x.construct_regression_data(kind=rul_type)
    optimizer = optim.Adam(cmapss_regressor.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-7)

    for epoch in range(epochs+1):
        cmapss_regressor.train()
        criterion = nn.MSELoss()
        for batch_id, (data, target) in enumerate(loader_train):
            optimizer.zero_grad()
            output = cmapss_regressor(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

class CMAPSSRNNRegressor(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, batch_size):
        super(CMAPSSRNNRegressor, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.hidden_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, x):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paper implementation for "Recurrent Neural Networks for Remaining Useful Life Estimation" by Felix O. Heimes.')
    parser.add_argument('task', type=str, help='Type of task to run (use "binary" for binary classification, "fcnn" for fully-connected network regression or "rnn" for RNN regression)')
    parser.add_argument('--data', type=str, help='Path to directory containing the dataset')
    parser.add_argument('--threshold', type=int, help='Number of first/last samples to be used as healthy/faulty samples (for binary classification only)')
    parser.add_argument('--rul-type', type=str, help='Type of RUL model to be used ("linear" for linearly decreasing RUL, "piecewise" for a piecewise function or "nonlinear" for a decreasing parabolic function)')
    parser.add_argument('--fd', type=int, default=1,  help='Specify which dataset variation to use (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--log-interval', type=int, default=4, help='Number of batches to wait before logging training status (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training (default: 64')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')

    args = parser.parse_args()

    if args.task == 'binary':
        run_cmapss_binary_classifier(args.data, args.fd, args.threshold, args.lr, args.batch_size, args.epochs, args.log_interval)

    if args.task == 'fcnn':
        run_cmapss_fcnn_regressor(args.data, args.fd, args.rul_type, args.lr, args.batch_size, args.epochs, args.log_interval)

model = run_cmapss_fcnn_regressor()

