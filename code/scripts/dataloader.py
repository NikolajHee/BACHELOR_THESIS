"""
Pytorch dataloader.
---
Maybe a bit too overkill, as the data can be loaded to memory without trouble
"""


import os
import pandas as pd
import wfdb
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/code/data/PTB_XL/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
#X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)




class PTB_XL(Dataset):
    def __init__(self, data_path, sampling_rate=100):
        """
        THIS DATASET IS INSPIRED BY THE 'example_physionet.py' PROVIDED BY THE DATA_COLLECTION TEAM OF PTB_XL

        
        data_path: path to folder PTB_XL/
        sampling_rate: amounts of sample per second
        """
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        
        self.df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'))

        self.paths = self.df.filename_lr

        self.agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)


    @staticmethod
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.paths[idx])
        
        signal, meta = wfdb.rdsamp(image_path)

        meta = pd.read_csv(os.path.join(path_to_data, 'ptbxl_database.csv'),  index_col='ecg_id')
        meta.scp_codes = meta.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load scp_statements.csv for diagnostic aggregation
        
        agg_df = agg_df[agg_df.diagnostic == 1]



        # Apply diagnostic superclass
        meta['diagnostic_superclass'] = meta.scp_codes.apply(aggregate_diagnostic)



        return signal, meta
    


