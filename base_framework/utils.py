"""
utils.py
- This script contains utility functions that are used in 
  the different scripts in the BACHELOR_THESIS folder.
"""


# imports
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random
import time



def random_seed(random_seed):
    """
    Function to seed the data-split and backpropagation (to enforce reproducibility)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed+1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed+2)
    random.seed(random_seed+3)


def save_parameters(save_path, dictionary):
    """
    Function to save parameters for reproducibility (wandb does this automatically)
    """
    with open(os.path.join(save_path, 'parameters.txt'), 'w') as f:
        for (key, value) in dictionary.items():
            f.write(f"{key} : {value}")
            f.write("\n")




def train_test_dataset(dataset, 
                       test_proportion, 
                       train_size=None, 
                       test_size=None, 
                       seed=None, 
                       return_stand=False):
    """
    Function to split the dataset into a training and test dataset.
    - dataset: the dataset to split
    - test_proportion: the proportion of the dataset to use for testing
    - train_size: the size of the training dataset
    - test_size: the size of the test dataset
    - seed: the seed for reproducibility
    - return_stand: whether to return standardize the data
    """

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # split indices
    train_indices, test_indices = train_test_split(np.arange(0, len(dataset)), test_size=test_proportion, random_state=seed)

    # create the datasets. 
    # The size of the datasets can be specified 
    #   (but must be less than the total size determined by the test_proportion)
    train_dataset = Subset(dataset, train_indices) if train_size is None else Subset(dataset, np.random.choice(train_indices, size=train_size, replace=False))
    test_dataset = Subset(dataset, test_indices) if test_size is None else Subset(dataset, np.random.choice(train_indices, size=test_size, replace=False))
    
    # get the shape of the data
    [T, D] = train_dataset[0][0].shape


    
    # standardize the data
    if return_stand:
        X = np.zeros((len(train_dataset), T, D))

        for i in range(len(train_dataset)):
            X[i,:,:] = train_dataset[i][0]

        # save in the class
        train_dataset.mean = np.mean(X)
        train_dataset.std = np.std(X)
        test_dataset.mean = np.mean(X)
        test_dataset.std = np.std(X)

    return train_dataset, test_dataset, D


def mia_train_test_dataset(in_dataset,
                           out_dataset,
                           N_train,
                           N_test,
                           seed):
    
   
    out_indices = out_dataset.indices[:N_test]
    in_indices = in_dataset.indices

    _N_out = len(out_indices) 
    _N_in = len(in_indices) 

    out_indices = np.random.choice(out_indices, size=_N_out, replace=False)
    in_indices = np.random.choice(in_indices, size=_N_in, replace=False)

    train_part_out = Subset(out_dataset.dataset, out_indices[:_N_out//2])
    train_part_in = Subset(in_dataset.dataset, in_indices[:_N_in//2])

    mia_train_data = combine(train_part_out, train_part_in, seed)

    test_part_out = Subset(out_dataset.dataset, out_indices[_N_out//2:])
    test_part_in = Subset(in_dataset.dataset, in_indices[_N_in//2:])

    mia_test_data = combine(test_part_out, test_part_in, seed)

    return mia_train_data, mia_test_data



from torch.utils.data import Dataset


class combine(Dataset):
    def __init__(self, data_1, data_2, seed):

        np.random.seed(seed)

        self.data_1 = data_1
        self.data_2 = data_2

        self.bin_indices = np.random.choice(np.r_[np.zeros(len(data_1)),  np.ones(len(data_2))], size=len(self), replace=False).astype(np.bool_)
        self._indices = np.zeros(len(self)).astype(np.int32)

        self._indices[~self.bin_indices] = np.arange(len(self.data_1))#.indices
        self._indices[self.bin_indices] = np.arange(len(self.data_2))#.indices

    
    def __len__(self):
        return len(self.data_1) + len(self.data_2)

    def __getitem__(self, idx):
        if self.bin_indices[idx]:
            return self.data_1[self._indices[idx]] + (0,)
        else:
            return self.data_2[self._indices[idx]] + (1,)





def baseline(train_dataset, test_dataset):
    """
    Calculate baseline given the training and test dataset.
        (Majority class prediction)
    """
    train = np.zeros(len(train_dataset))
    test = np.zeros(len(test_dataset))

    # get the train labels
    for i in range(len(train_dataset)):
        train[i] = train_dataset[i][1][0]

    # get the test labels
    for i in range(len(test_dataset)):
        test[i] = test_dataset[i][1][0]

    # get the unique labels and their counts
    unique_labels, count = np.unique(train, return_counts=True)

    # get the most common label
    index = np.argmax(count)

    # mean accuracy
    baseline = np.mean(unique_labels[index] == test)

    return baseline
        


def cool_plots():
    """
    Function to set the style of the plots to the science style.
    """
    import scienceplots
    import matplotlib.pyplot as plt
    plt.style.use('science')
    plt.rcParams.update({'figure.dpi': '200'})
    plt.rcParams.update({"legend.frameon": True})



class TimeTaking:
    """
    Small class to keep track of time for different tasks.
    """
    def __init__(self, save_path, verbose=False):
        self.start_time = {}
        self.end_time = {}
        self.pause_time = {}
        self.save_path = save_path
        self.verbose = verbose
        self._pause = False

    def start(self, name):
        if name not in self.pause_time.keys():
            if self.verbose: print(f"Starting {name}.")
            self.start_time[name] = time.time()
        else:
            self.start_time[name] += time.time() - self.pause_time.pop(name)

    def pause(self, name):
        self.pause_time[name] = time.time()

    def end(self, name):
        if name in self.pause_time.keys():
            self.start_time[name] += time.time() - self.pause_time.pop(name)
        if self.verbose: print(f"Ending {name}.")
        self.end_time[name] = time.time()

    def save(self):
        with open('time.txt', 'w') as f:
            for (key, value) in self.start_time.items():
                f.write(f"{key} : {self.start_time[key]} - {self.end_time[key]} = {self.end_time[key] - self.start_time[key]}s")
                f.write("\n")

    def output_dict(self, key):
        if key not in self.pause_time.keys():
            save = {key: time.time() - self.start_time[key]}
        else:
            save = {key: self.pause_time[key] - self.start_time[key]}

        return save
    
    def log_wandb(self, wandb, key):
        save = self.output_dict(key)
        wandb.log(save)
    


    def pass_to_wandb(self, wandb):
        for (key, value) in self.start_time.items():
            wandb.log({key: self.end_time[key] - self.start_time[key]})
            
def remove_list(args):
    """
    DEPRICATED FUNCTION.
    """
    #save = {}
    dict_ = vars(args)
    for (i,j) in dict_.items():
        if type(j) == list:
            args[i] = j[0]
        else:
            args[i] = j
    #return save

class random_cropping:
    """
    DEPRICATED FUNCTION.

    Cropping at random
        works only for 1 dimensional right now?
        Only used in training
    """
    def __init__(self, verbose=False):
        self.verbose = verbose


    def __call__(self, x):
        [T, D] = x.shape

        values, indices  = torch.randint(T, (1,4)).sort(1)


        a1, a2, b1, b2 = (values[:,i] for i in range(4))


        if self.verbose:
            print(f"cropping between {a1, b1} and {a2, b2}")

        return (x[a1:b1, :], 
                x[a2:b2, :], 
                {'a1': a1.item(), 'a2': a2.item(), 'b1': b1.item(), 'b2': b2.item()})


# class CNN_block(nn.Module):
#     """
#     dumped
#     """
#     def __init__(self, l, dim):
#         super().__init__()
#         self.conv1 = nn.Conv1d(dim, dim, padding='same', kernel_size=3, dilation=2**l)
#         self.conv2 = nn.Conv1d(dim, dim, padding='same', kernel_size=3, dilation=2**l)

#     def forward(self, x):
#         residual = x
#         x = self.conv1(x)
#         x = nn.ReLU()(x)
#         x = self.conv2(x)

#         return x + residual


# class dilated_CNN_layer(nn.Module):
#     """
#     dumped
#     """
#     """
#     Hmm a bit difficult.

#     Ten residual blocks
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.CNN = nn.Sequential(*[
#             CNN_block(l=i, dim=dim) 
#             for i in range(10)
#         ])

    
#     def __call__(self, x):
#         # input x is (batch_size, T, dimension)
#         return self.CNN(x)


if __name__ =='__main__':


    from base_framework.dataset import PTB_XL
    dataset = PTB_XL('BACHELOR_THESIS/BACHELOR_THESIS/PTB_XL')



    train_dataset, test_dataset, D = train_test_dataset(dataset=dataset,
                                                        test_proportion=0.3,
                                                        train_size=3000,
                                                        test_size=2000,
                                                        seed=1,
                                                        return_stand=False)  

    test1, test2 = mia_train_test_dataset(out_dataset=test_dataset, in_dataset=train_dataset, N_train=None, N_test=None, seed=1)       


    print('abe')


