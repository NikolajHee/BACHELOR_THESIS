"""
Pytorch dataloaders.
    - PTB_XL
    - AEON_DATA
        (UCR and UEA datasets)
"""

# imports
import os
import ast
import wfdb
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from aeon.datasets import load_classification


class PTB_XL(Dataset):
    def __init__(self, data_path=None, sampling_rate=100, multi_label=False):
        """
        THIS DATASET IS INSPIRED BY THE 'example_physionet.py' PROVIDED BY THE DATA_COLLECTION TEAM OF PTB_XL

        
        data_path: path to folder PTB_XL/
        sampling_rate: amounts of sample per second (Not used)
        """
        # load the data (if no path is given, load the path from the save_path.pkl file)
        if data_path is None:
            with open('save_path.pkl', 'rb') as fp:
                data_path = pickle.load(fp)['data_path']

        self.data_path = data_path
        self.sampling_rate = sampling_rate # not used

        # initialize mean and std (doesn't normalize the data if not changed)
        self.mean, self.std = None, None

        # load the metadata
        self.df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'))

        # convert the strings to actual dictionaries
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # load the paths
        self.paths = self.df.filename_lr

        # load the diagnostic classes
        agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)

        # remove points that are actually not in the dataset
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(aggregate_diagnostic)

        # create binary label (1 if NORM, 0 else)
        self.df['binary_label'] = [1 - (i == ['NORM']) for i in self.df.diagnostic_superclass]

        # self.test()
        self.ml = False
        if multi_label:
            self.ml = True
            self.multi_label()
        

    
    def test(self):
        """
        The function is used to create a class for each combination of diagnostic_superclass
            Only for testing if the representation space has learned the classes
        """
        save = ["" for _ in range(len(self.df.diagnostic_superclass))]

        for i, element in enumerate(self.df.diagnostic_superclass):
            if len(element) == 0:
                save[i] = 'Empty'
            else:
                for el in element:
                    save[i] += el

        le = LabelEncoder()
        test = le.fit_transform(save)
        self.df['test'] = test

    def multi_label(self):
        test = np.zeros((len(self.df), 5))

        self.classes = ['NORM', 'STTC', 'MI', 'HYP', 'CD']

        for i, el in enumerate(self.df.diagnostic_superclass):
            test[i] = [1 if c in el else 0 for c in self.classes]
        
        for i, c in enumerate(self.classes):
            self.df[c] = test[:,i]
    


    def normalize(self, X):
        # normalize the data
        if self.mean and self.std:
            return (X - self.mean) / self.std
        return X
    

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        print(idx)
        if type(idx) is np.ndarray:
            idx = list(idx)

        if type(idx) is slice:
            # get the signal and label
            signal = np.stack([self.normalize(wfdb.rdsamp(os.path.join(self.data_path, path))[0]) for path in self.paths[idx]])
            if not self.ml:
                label = self.df.binary_label[idx]
            else:
                label = self.df.iloc[idx,-5:]
            
            #label = self.df['test'][idx]
        elif (type(idx) is tuple) or (type(idx) is list):
            # get the signal and label
            signal = np.stack([self.normalize(wfdb.rdsamp(os.path.join(self.data_path, self.paths[i]))[0]) for i in idx])
            if not self.ml:
                label = [self.df.binary_label[i] for i in idx]
            else:
                label = [self.df.iloc[i,-5:] for i in idx]
            
            #label = [self.df['test'][i] for i in idx]
        else:
            # get the signal and label
            image_path = os.path.join(self.data_path, self.paths[idx])
            signal = self.normalize(wfdb.rdsamp(image_path)[0])
            if not self.ml:
                label = self.df.binary_label[idx]
            else:
                label = self.df.iloc[idx,-5:]
            
            #label = self.df['test'][idx]
        
        return torch.tensor(signal), torch.tensor(label)




class AEON_DATA(Dataset):
    """
    AEON_DATA is a dataset class for UCR and
    UEA datasets. The data is loaded using the  
    aeon.datasets.load_classification function.
    """
    def __init__(self, name):
        self.mean, self.std = None, None

        self.X, self.y = load_classification(name)

        if type(self.y[0]) is np.str_:
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)

    
    def __len__(self):
        return len(self.y)
    
    def normalize(self, X):
        # normalize the data
        if self.mean and self.std:
            return (X - self.mean) / self.std
        return X

    def __getitem__(self, idx):
        return self.normalize(self.X[idx].T), self.y[idx]
    

if __name__ == '__main__':        
    data = PTB_XL('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/code/data/PTB_XL', multi_label=True)

    #print(data[0:10])

    loader = DataLoader(data, batch_size=32, shuffle=True)

    print('test')

    print(next(iter(loader)))

    #print(data[1,10,20])

