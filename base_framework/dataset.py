"""
Pytorch dataloaders.
    * PTB_XL
    * AEON_DATA
        (UCR and UEA datasets)
    * PTB_XL_v2
    * AEON_DATA_v2
"""

# imports
import os
import ast
import wfdb
import torch
import pickle
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


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
        self.df['binary_label'] = [1 - ('NORM' in i) for i in self.df.diagnostic_superclass]

        # self.test()
        self.ml = False
        if multi_label:
            self.ml = True
            self.multi_label()
        
    def multi_label(self):
        self.classes = ['NORM', 'STTC', 'MI', 'HYP', 'CD']

        for i, c in enumerate(self.classes):
            self.df[c] = self.df['diagnostic_superclass'].map(lambda x: int(c in x))

    def normalize(self, X):
        # normalize the data
        if self.mean and self.std:
            return (X - self.mean) / self.std
        return X
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #print(idx)
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
        
        return torch.tensor(signal), (torch.tensor(label), )



class PTB_XL_v2(PTB_XL):
    def __init__(self, data_path=None, sampling_rate=100, multi_label=False):
        super().__init__(data_path=data_path,
                         sampling_rate=sampling_rate,
                         multi_label=multi_label)


    def __getitem__(self, idx):
        X, y = super().__getitem__(idx) 

        return (X, y + (idx, ))
    


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

        self.N_classes = len(np.unique(self.y))

    
    def __len__(self):
        return len(self.y)
    
    def normalize(self, X):
        # normalize the data
        if self.mean and self.std:
            return (X - self.mean) / self.std
        return X

    def __getitem__(self, idx):
        return torch.tensor(self.normalize(self.X[idx].T)), (torch.tensor(self.y[idx]),)


class AEON_DATA_v2(AEON_DATA):
    def __init__(self, name):
        super().__init__(name)
    
    def __getitem__(self, idx):
        X, y = super().__getitem__(idx) 
        return (X, y + (idx, ))
    

class mia_train(Dataset):
    def __init__(self, 
                 train_dataset, 
                 test_dataset):
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    def __len__(self):
        return len(self.train_dataset) + len(self.test_dataset)

    def __getitem__(self, idx):
        if len(self.train_dataset) > idx:
            return self.train_dataset[idx] + (1,)
        if len(self.train_dataset) <= idx:
            return self.test_dataset[idx - len(self.train_dataset)] + (0,)

class mia_data(Dataset):
    def __init__(self, 
                 dataset,
                 label):
        self._dataset = dataset
        self._label = label
    
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx] + (self.label,)



class shuffle(Dataset):
    def __init__(self, dataset, seed):
        np.random.seed(seed)

        self.data = dataset

        self._indices = np.random.choice(self.data.indices, size=len(self.data.indices), replace=False)


    def __getitem__(self, idx):
        return self.data[self._indices[idx]]




if __name__ == '__main__':        
    data = PTB_XL_v2('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/PTB_XL', multi_label=True)

    #print(data[0:10])

    ys = np.zeros((len(data), 5))
    for i in range(len(data)):
        ys[i] = data[i][1]

    loader = DataLoader(data, batch_size=32, shuffle=False)

    

    for (train, test, index) in loader:
        print('test')

    print(next(iter(loader)))

    #print(data[1,10,20])

