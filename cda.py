"""
CCA.py
"""


# import cca
from sklearn.cross_decomposition import CCA
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os


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


    # get output dimension and batch size
    output_dim = model.module.output_dim
    batch_size = train_loader.batch_size
    train_batches, test_batches = len(train_loader), len(test_loader)

    # initialize
    Z_train = np.zeros((batch_size * train_batches, output_dim))
    Y_train = np.zeros((batch_size * train_batches, 5))
    Z_test= np.zeros((batch_size * test_batches, output_dim))
    Y_test = np.zeros((batch_size * test_batches, 5))

    # train loop to convert all data to features
    for i, (X, y) in enumerate(train_loader):
        z = model(X.to(device).float()) # output: N x T x Dr
        
        
        z = z.transpose(1,2) # N x Dr x T

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2]) # N x Dr

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

        y = y.numpy()

        Z_train[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_train[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)

    # Perform CCA
    model_ = CCA(n_components=2)

    model_.fit(Z_train, Y_train) 

    # extract the basis vectors
    train_vector_x, train_vector_y = model_.x_loadings_, model_.y_loadings_

    # transform the data
    train_x, train_y = model_.transform(Z_train, Y_train)

    # test loop to convert all data to features
    for i, (X, y) in enumerate(test_loader):
        z = model(X.to(device).float())

        z = z.transpose(1,2) # N x Dr x T

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        y = y.numpy()

        Z_test[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_test[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)


    # transform the data
    test_x, test_y = model_.transform(Z_test, Y_test)

    # save the data
    np.save(os.path.join(save_path, 'train_x.npy'), train_x)
    np.save(os.path.join(save_path, 'train_y.npy'), train_y)
    np.save(os.path.join(save_path, 'train_vector_x.npy'), train_vector_x)
    np.save(os.path.join(save_path, 'train_vector_y.npy'), train_vector_y)
    np.save(os.path.join(save_path, 'test_x.npy'), test_x)
    np.save(os.path.join(save_path, 'test_y.npy'), test_y)



