"""
Clustering.py
    - This script is used to cluster the extracted features from the model using t-SNE.
"""

# imports
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


# main function
def tsne(H, 
         train_dataloader:DataLoader,
         test_dataloader:DataLoader,
         device):
    """
    H : encoder to create representations
    train_loader : dataloader for the training split
    test_loader : dataloader for the test split
    output_dim : the output dimension of the representations (could maybe be infered from H)
    device : the torch_device where the models resides. 
    """
    # get output dimension and batch size
    output_dim = H.module.output_dim
    batch_size = train_dataloader.batch_size
    train_batches, test_batches = len(train_dataloader), len(test_dataloader)

    # initialize
    Z_train = np.zeros((batch_size * train_batches, output_dim))
    Z_test= np.zeros((batch_size * test_batches, output_dim))
    Y_train = np.zeros((batch_size * train_batches))
    Y_test = np.zeros(batch_size * test_batches)

    # get train representations
    for i, (X, y) in tqdm(enumerate(train_dataloader)):    
        z = H.forward(X.float().to(device))

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        #  TODO: What about TSNE though???
        z = z.transpose(1,2) # N x Dr x T

        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        if type(y) is tuple:
            y = np.array(y)
        else:
            y = y.numpy()

        Z_train[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_train[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)

    # perform t-SNE
    model_ = TSNE(n_components=2)
    self_organized_map = model_.fit_transform(Z_train)

    # plot the t-SNE
    fig_train, ax1 = plt.subplots()

    classes = np.unique(Y_train)
    for _class in classes:
        index = Y_train == _class
        ax1.plot(self_organized_map[index,0], self_organized_map[index,1], '.', label = str(_class))
    plt.legend()
    plt.close()

    # get test representations
    for i, (X, y) in tqdm(enumerate(test_dataloader)):
        z = H.forward(X.to(device).float())

        z = z.transpose(1,2) # N x Dr x T

        # Maxpooling is inspried by the TS2VEC framework for classification
        # do they also do this when they are plotting tsne?
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        
        if type(y) is tuple:
            y = np.array(y)
        else:
            y = y.numpy()

        Z_test[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_test[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)

    # TODO: transform or fit_transform?
    # perform t-SNE
    self_organized_map = model_.fit_transform(Z_test)

    # plot the t-SNE
    fig_test, ax2 = plt.subplots()

    classes = np.unique(Y_test)
    for _class in classes:
        index = Y_test == _class
        ax2.plot(self_organized_map[index,0], self_organized_map[index,1], '.', label = str(_class))
    plt.legend()
    plt.close()

    return fig_train, fig_test




