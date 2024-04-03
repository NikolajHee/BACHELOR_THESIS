"""
Pytorch dataloader.
"""
import os
import pandas as pd
import wfdb
from torch.utils.data import Dataset
import pandas as pd
import wfdb
import ast
import pickle
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split



class PTB_XL(Dataset):
    def __init__(self, data_path=None, sampling_rate=100, binary_label:bool=True):
        """
        THIS DATASET IS INSPIRED BY THE 'example_physionet.py' PROVIDED BY THE DATA_COLLECTION TEAM OF PTB_XL

        
        data_path: path to folder PTB_XL/
        sampling_rate: amounts of sample per second
        """
        if data_path is None:
            path = '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/code/scripts/save_path.pkl'
            with open('code/scripts/save_path.pkl', 'rb') as fp:
            #with open(path, 'rb') as fp:
                data_path = pickle.load(fp)['data_path']

        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.binary_label = binary_label
        
        self.df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'))
        # convert the strings to actual dictionaries
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))

        self.paths = self.df.filename_lr

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
        self.df['binary_label'] = [1 - (i == ['NORM']) for i in self.df.diagnostic_superclass]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.paths[idx])
        
        signal, meta = wfdb.rdsamp(image_path)

        label = self.df.binary_label[idx]

        return signal, label
    

def train_test_loaders(dataset, batch_size, test_size, verbose=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
    train_indices, test_indices = train_test_split(np.arange(0, len(dataset)), test_size=test_size)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if verbose:
        print(f"train batches {len(train_dataloader)} ({len(train_dataloader.dataset)} observations).")
        print(f"test batches {len(test_dataloader)} ({len(test_dataloader.dataset)} observations).")
    
    return train_dataloader, test_dataloader



if __name__ == '__main__':        
    data = PTB_XL()

    train_loader, test_loader = train_test_loaders(data, 16, 0.1)

    pass
    


    


