"""
CCA.py
"""


# import cca
from sklearn.cross_decomposition import CCA
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import torch

def cca(model, 
        train_loader,
        test_loader:DataLoader,
        device,
        save_path:str):
    """
    Canonical Correlation Analysis (CCA) function.

    Should be used to extract features from the data and then use
    CCA to examine wether or not the features are correlated with
    the multi-labels of PTB_XL.
    """
    print('Running CCA.')

    # get output dimension and batch size
    output_dim = model.output_dim
    batch_size = train_loader.batch_size
    train_batches, test_batches = len(train_loader), len(test_loader)

    # initialize
    Z_train = np.zeros((batch_size * train_batches, output_dim))
    Y_train = np.zeros((batch_size * train_batches, 5))
    Z_test= np.zeros((batch_size * test_batches, output_dim))
    Y_test = np.zeros((batch_size * test_batches, 5))

    print('Collecting matrices.')
    # train loop to convert all data to features
    for i, (X, y) in enumerate(train_loader):
        z = model.model(X.to(device).float()) # output: N x T x Dr
        
        
        z = z.transpose(1,2) # N x Dr x T

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2]) # N x Dr

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

        y = y.numpy()

        Z_train[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_train[i*batch_size:(i+1)*batch_size, :] = y.reshape(batch_size, 5)

    # Perform CCA
    print('fitting CCA')
    model_ = CCA(n_components=2)

    model_.fit(Z_train, Y_train) 


    # extract the basis vectors
    train_vector_x, train_vector_y = model_.x_loadings_, model_.y_loadings_

    print('transforming data.')
    # transform the data
    train_x, train_y = model_.transform(Z_train, Y_train)

    print('Collecting matrices.')
    # test loop to convert all data to features
    for i, (X, y) in enumerate(test_loader):
        z = model.model(X.to(device).float())

        z = z.transpose(1,2) # N x Dr x T

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        y = y.numpy()

        Z_test[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_test[i*batch_size:(i+1)*batch_size, :] = y.reshape(batch_size, 5)
    print('fitting CCA')
    model_ = CCA(n_components=2)
    model_.fit(Z_train, Y_train) 

    test_vector_x, test_vector_y = model_.x_loadings_, model_.y_loadings_

    # transform the data
    print('transforming data.')
    test_x, test_y = model_.transform(Z_test, Y_test)

    # save the data
    np.save(os.path.join(save_path, 'train_x.npy'), train_x)
    np.save(os.path.join(save_path, 'train_y.npy'), train_y)
    np.save(os.path.join(save_path, 'train_vector_x.npy'), train_vector_x)
    np.save(os.path.join(save_path, 'train_vector_y.npy'), train_vector_y)
    np.save(os.path.join(save_path, 'test_x.npy'), test_x)
    np.save(os.path.join(save_path, 'test_y.npy'), test_y)
    np.save(os.path.join(save_path, 'test_vector_x.npy'), test_vector_x)
    np.save(os.path.join(save_path, 'test_vector_y.npy'), test_vector_y)



if __name__ == '__main__':
    
    # set random seed
    from utils import save_parameters, random_seed, TimeTaking

    random_seed(0)


    from dataset import PTB_XL
    dataset = PTB_XL(multi_label=True)



    # create train/test-split
    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                     test_proportion=0.3,
                                                     train_size=None,
                                                     test_size=None,
                                                     return_stand=False)
    
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    from TS2VEC import TS2VEC

    model = TS2VEC(12, 32, 320, 0.5, DEVICE)

    path = 'results/PTB_XL/(12_05_2024)_(23_52_29)/model.pt'

    model.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    cca(model, 
        train_loader=train_dataloader, 
        test_loader=test_dataloader, 
        device=DEVICE, 
        save_path='')
                                                 