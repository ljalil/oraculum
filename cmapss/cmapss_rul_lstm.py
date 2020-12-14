import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import importlib
import cmapssdataset

#%%

class CMAPSSLSTMRUL(nn.Module):
    def __init__(self):
        super(CMAPSSLSTMRUL, self).__init__()

