

import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np


def save_parameters(save_path, dictionary):
    with open(os.path.join(save_path, 'parameters.txt'), 'w') as f:
        for (key, value) in dictionary.items():
            f.write(f"{key} : {value}")
            f.write("\n")





def train_test_dataset(dataset, test_proportion, train_size=None, test_size=None, verbose=False, seed=None, ):

    if seed is not None:
        np.random.seed(seed)
    train_indices, test_indices = train_test_split(np.arange(0, len(dataset)), test_size=test_proportion)
    if (train_size is None) and (test_size is None):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
    else:
        train_dataset = Subset(dataset, train_indices[:train_size])
        test_dataset = Subset(dataset, test_indices[:test_size])

    return train_dataset, test_dataset

def baseline(train_dataset, test_dataset):

    train = np.zeros(len(train_dataset))
    test = np.zeros(len(test_dataset))
    for i in range(len(train_dataset)):
        train[i] = train_dataset[i][1]
    for i in range(len(test_dataset)):
        test[i] = test_dataset[i][1]

    test1, test2 = np.unique(train, return_counts=True)

    index = np.argmax(test2)

    baseline = np.mean(test1[index] == test)

    return baseline
        


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

if __name__ =='__main__':
    save_parameters('', test='hej', test2='hej2', test3='hej4')

