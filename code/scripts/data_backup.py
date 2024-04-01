import os
import pandas as pd
import numpy as np
import wfdb
import torch
from random import shuffle
import random
import ast


random.seed(0)
print("data-shuffling is seeded.")


class PTB_XL:
    def __init__(self, 
                 batch_size, 
                 path_to_data='/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/code/data/PTB_XL',
                 shuffle_=True
                 ):
        self.batch_size = batch_size
        self.path_to_data = path_to_data

        meta = pd.read_csv(os.path.join(path_to_data, 'ptbxl_database.csv'),  index_col='ecg_id')
        meta.scp_codes = meta.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path_to_data + '/scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass
        meta['diagnostic_superclass'] = meta.scp_codes.apply(aggregate_diagnostic)


        [n1, p1] = meta.shape

        index = np.arange(n1)

        if shuffle_:
            shuffle(index)

        self.filenames = iter(meta.filename_lr.iloc[index])
        self.labels = iter(meta.diagnostic_superclass.iloc[index])



    def load_some_signals(self):
        """
        path_to_data : self-explainatory

        """
        save = []
        labels = []

        for i in range(self.batch_size):
            
            xtest, meta_ = wfdb.rdsamp(os.path.join(self.path_to_data,next(self.filenames)))
            label = next(self.labels)
            labels.append(label)
            save.append(xtest)
            

        
        sick_or_healthy = [1 - (i == ['NORM']) for i in labels]
        sick_or_healthy = torch.tensor(sick_or_healthy)
        return torch.stack([torch.from_numpy(i).float() for i in save]), sick_or_healthy



if __name__ == '__main__':
    save = PTB_XL(100)

    print(save.load_some_signals())
        