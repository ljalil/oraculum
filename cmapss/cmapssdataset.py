import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
#%%
class CMAPSSDataset():
    def __init__(self, path_to_data, fd_number):
        self.columns = ['unit','cycle','op1','op2','op3']+['sensor_{}'.format(str(i).zfill(2)) for i in range(1,24)]
        self.sensors = ['Total temperature at fan inlet',
           'Total temperature at LPC outlet',
           'Total temperature at HPC outlet',
           'Total temperature at LPT',
           'P2 Pressure at fan inlet',
           'Total pressure in bypass-duct',
           'Total pressure at HPC outlet',
           'Physical fan speed',
           'Physical core speed',
           'Engine pressure ratio (P50/P2)',
           'Static pressure at HPC outlet',
           'Ratio of fuel flow to Ps30',
           'Corrected fan speed',
           'Corrected core speed',
           'Bypass Ratio',
           'Burner fuel-air ratio',
           'Bleed Enthalpy',
           'Demanded fan speed',
           'Demanded corrected fan speed',
           'HPT coolant bleed',
           'LPT coolant bleed']

        self.path_to_file  = os.path.join(path_to_data, 'train_FD00'+str(fd_number)+'.txt')
        self.data = pd.read_csv(self.path_to_file, sep=' ', names=self.columns)
        self.data = self.data.drop(['sensor_22','sensor_23'], axis=1)
        self.rul = np.array([])
        self.random_test_units = np.random.uniform(1, self.data['unit'].max(), 4).astype('int32')

        self.sequence_length = 50
        self.start_of_failure = 120
        self.constant_rul = 130

    def RUL_modeler(self, life : int, kind : str, classes = 5):
        """Constructs Remaining Useful Life (RUL) according to different models"""
        if kind == 'linear':
            return np.flip(np.arange(life)).reshape(-1,1)

        elif kind == 'nonlinear':
            x = np.array([0, 1.4*life/2, life])
            y = np.array([life, 1.2*life/2, 0])
            polynom = np.poly1d(np.polyfit(x,y,3))
            return polynom(np.arange(0,life)).reshape(-1,1)

        elif kind == 'piecewise':
            x = np.concatenate(( self.constant_rul*np.ones((self.start_of_failure,1)), np.linspace(self.constant_rul,0,life-self.start_of_failure).reshape(-1,1)), axis=0)
            return x

        elif kind == 'classification':
            return np.linspace(0, classes, life, endpoint = False ).astype('int32' ).reshape(-1,1)

    def calculate_RUL_all_units(self, kind : str):
        self.rul = np.array([])
        all_units_lengths = self.data.groupby('unit')['cycle'].max()

        for unit_length in all_units_lengths:
            self.rul = np.append(self.rul, self.RUL_modeler(unit_length, kind))

    def calculate_RUL_from_dataframe(self, df, kind : str):
        df_rul = np.array([])
        all_units_lengths = df.groupby('unit')['cycle'].max()

        for unit_length in all_units_lengths:
            df_rul = np.append(df_rul, self.RUL_modeler(unit_length, kind))

        return df_rul

    def construct_binary_classification_data(self, good_faulty_threshold = 30, batch_size = 64):
        x = np.zeros((0,26))
        y = np.zeros((0,1))

        for group_name, group_df in self.data.groupby('unit'):
            first_rows = group_df[:good_faulty_threshold]
            last_rows = group_df[-good_faulty_threshold:]

            x = np.concatenate((x, first_rows, last_rows), axis=0)
            y = np.concatenate((y, np.zeros((good_faulty_threshold,1)), np.ones((good_faulty_threshold,1))), axis=0)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        tensor_x_train = torch.Tensor(x_train)
        tensor_x_test = torch.Tensor(x_test)
        tensor_y_train = torch.Tensor(y_train)
        tensor_y_test = torch.Tensor(y_test)

        loader_train = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=batch_size)
        loader_test = DataLoader(TensorDataset(tensor_x_test, tensor_y_test), batch_size=batch_size)
        return loader_train, loader_test

    def dataset_statistics(self):
        units_number = int(self.data.iloc[-1]['unit'])
        max_length = self.data.groupby('unit')['cycle'].max().max()
        average_length = around(self.data.groupby('unit')['cycle'].max().mean(),2)
        min_length = self.data.groupby('unit')['cycle'].max().min()

        return dict({'units' : units_number, 'max': max_length, 'average' : average_length, 'min':min_length})

    def construct_regression_data(self, batch_size=64,  kind='nonlinear'):
        x_train_fcnn = self.data.copy()
        x_test_fcnn = pd.DataFrame(columns = self.data.columns)

        for test_unit in self.random_test_units:
            cond = x_train_fcnn.unit == test_unit 
            x_test_fcnn = pd.concat((x_test_fcnn, x_train_fcnn[cond]), axis=0)

        x_train_fcnn.drop(x_test_fcnn.index, axis=0, inplace=True)

        y_train_fcnn = self.calculate_RUL_from_dataframe(x_train_fcnn, kind=kind)
        y_test_fcnn = self.calculate_RUL_from_dataframe(x_test_fcnn, kind=kind)
        y_train_fcnn = y_train_fcnn.reshape(-1,1)
        y_test_fcnn = y_test_fcnn.reshape(-1,1)
        x_train_fcnn = x_train_fcnn.values[:, 2:]
        x_test_fcnn = x_test_fcnn.values[:, 2:]
        tensor_x_train_fcnn = torch.Tensor(x_train_fcnn.astype('float32'))
        tensor_x_test_fcnn = torch.Tensor(x_test_fcnn.astype('float32'))
        tensor_y_train_fcnn = torch.Tensor(y_train_fcnn)
        tensor_y_test_fcnn = torch.Tensor(y_test_fcnn)
        loader_train_fcnn = DataLoader(TensorDataset(tensor_x_train_fcnn, tensor_y_train_fcnn), shuffle=True, batch_size=batch_size)
        loader_test_fcnn = DataLoader(TensorDataset(tensor_x_test_fcnn, tensor_y_test_fcnn), shuffle=True, batch_size=batch_size)

        return loader_train_fcnn, loader_test_fcnn
 
    def get_random_test_unit(self, id: int):
        return self.data[self.data['unit'] == self.random_test_units[id]]

cm = CMAPSSDataset('~/Workspace/Datasets/CMAPSS/',1)
