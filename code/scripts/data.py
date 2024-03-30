import os
import pandas as pd
import numpy as np
import wfdb
import torch
from random import shuffle
import random


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

        [n1, p1] = meta.shape

        if shuffle_:
            keys = list(meta.filename_lr.keys())

            shuffle(keys)

            self.filenames = iter(meta.filename_lr[keys])
        else:
            self.filenames = iter(meta.filename_lr)

    def load_some_signals(self):
        """
        path_to_data : self-explainatory

        """
        save = []

        for i in range(self.batch_size):
            xtest, meta_ = wfdb.rdsamp(os.path.join(self.path_to_data,next(self.filenames)))
            save.append(xtest)


        return torch.stack([torch.from_numpy(i).float() for i in save])



if __name__ == '__main__':
    save = PTB_XL(10)

    print(save.load_some_signals())
        