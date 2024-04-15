"""
Clustering.py
"""

# imports
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import os


# main function
def tsne(H, 
         train_dataset,
         test_dataset,
         output_dim,
         device,
         save_path):
    """
    H : encoder to create representations
    train_loader : dataloader for the training split
    test_loader : dataloader for the test split
    output_dim : the output dimension of the representations (could maybe be infered from H)
    device : the torch_device where the models resides. 
    save_path : where to save the png's
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    batch_size = train_dataloader.batch_size
    train_batches, test_batches = len(train_dataloader), len(test_dataloader)

    Z_train = np.zeros((batch_size * train_batches, output_dim))
    Y_train = np.zeros((batch_size * train_batches))

    Z_test= np.zeros((batch_size * test_batches, output_dim))
    Y_test = np.zeros(batch_size * test_batches)


    for i, (X, y) in tqdm(enumerate(train_dataloader)):    
        z = H.forward(X.float().to(device))

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        #  
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        if type(y) is tuple:
            y = np.array(y)
        else:
            y = y.numpy()

        Z_train[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_train[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)
        #classes_train[i*batch_size:(i+1)*batch_size] = 

    tsne = TSNE(n_components=2).fit_transform(Z_train)

    fig_train, ax1 = plt.subplots()

    classes = np.unique(Y_train)
    for _class in classes:
        index = Y_train == _class
        ax1.plot(tsne[index,0], tsne[index,1], '.', label = str(_class))#, c=Y_test)
     
    #plt.legend()
    #wandb.log({"train_t_sne": fig})
    plt.legend()
    plt.savefig(os.path.join(save_path, 'train_t_sne.png'))
    plt.close()

    for i, (X, y) in tqdm(enumerate(test_dataloader)):
        z = H.forward(X.to(device).float())

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

    
    tsne = TSNE(n_components=2).fit_transform(Z_test)

    fig_test, ax2 = plt.subplots()

    classes = np.unique(Y_test)
    for _class in classes:
        index = Y_test == _class
        ax2.plot(tsne[index,0], tsne[index,1], '.', label = str(_class))#, c=Y_test)

    #plt.legend()
        

    #wandb.log({"test_t_sne": fig})
    plt.legend()
    plt.savefig(os.path.join(save_path, 'test_t_sne.png'))
    plt.close()

    return fig_train, fig_test


def main():
    np.random.seed(0)


    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    from TS2VEC import TS2VEC
    model_ = TS2VEC(input_dim=12, 
                   output_dim=320, 
                   hidden_dim=64,
                   p=0.5, 
                   device=DEVICE, 
                   verbose=True).to(DEVICE)
    
    model = torch.optim.swa_utils.AveragedModel(model_)

    PATH = '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/results/maybe_works/(08_04_2024)_(21_03_13)/best_model.pt'
    PATH = 'results/maybe_works/(08_04_2024)_(21_03_13)/best_model.pt'

    #print([test[0] for test in test_])
    model.load_state_dict(torch.load(PATH))
    model.eval()

    from BACHELOR_THESIS.dataset import PTB_XL
    dataset = PTB_XL()

    N_train, N_test, verbose = 5000, 500, True

    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                        test_proportion=0.1,
                                                        train_size=N_train,
                                                        test_size=N_test,
                                                        verbose=verbose,
                                                        seed=0)

    batch_size = 64

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    tsne(model, train_dataloader, test_dataloader, output_dim=320, device=DEVICE)
    


def main2():
    

    
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    from TS2VEC import TS2VEC
    model_ = TS2VEC(input_dim=1, 
                   output_dim=320, 
                   hidden_dim=64,
                   p=0.5, 
                   device=DEVICE, 
                   verbose=True).to(DEVICE)

    model = torch.optim.swa_utils.AveragedModel(model_)
    

    from BACHELOR_THESIS.dataset import PTB_XL
    dataset = PTB_XL()

    N_train, N_test, verbose = 10, 10, True

    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                        test_proportion=0.3,
                                                        train_size=N_train,
                                                        test_size=N_test,
                                                        verbose=verbose,
                                                        seed=0)


    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, drop_last=True)
    
    
    tsne(model, train_dataloader, test_dataloader, output_dim=320, device=DEVICE)






if __name__ == '__main__':

    main2()