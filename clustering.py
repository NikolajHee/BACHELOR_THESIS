from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch


def tsne(H, 
         train_loader,
         test_loader,
         device):


    batch_size = train_loader.batch_size
    train_batches, test_batches = len(train_loader), len(test_loader)

    Z_train = np.zeros((batch_size * train_batches, H.output_dim))
    Y_train = np.zeros((batch_size * train_batches))
    Z_test= np.zeros((batch_size * test_batches, H.output_dim))
    Y_test = np.zeros(batch_size * test_batches)

    
    for i, (X, y) in tqdm(enumerate(train_loader)):
        print(X)
        z = H(X.to(device).float())

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

        y = y.numpy()

        Z_train[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, H.output_dim)
        Y_train[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)

    test1 = TSNE(n_components=2).fit_transform(Z_train)

    fig, ax = plt.subplots()

    classes = np.unique(Y_train)
    for _class in classes:
        index = Y_train == _class
        ax.plot(test1[index,0], test1[index,1], '.', label = str(_class))#, c=Y_test)
     
    #plt.legend()
    #wandb.log({"train_t_sne": fig})

    plt.savefig('train_t_sne.png')
    plt.close()

    for i, (X, y) in tqdm(enumerate(test_loader)):
        z = H(X.to(device).float())

        # Maxpooling is inspried by the TS2VEC framework for classification
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        y = y.numpy()

        Z_test[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, H.output_dim)
        Y_test[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)

    
    test2 = TSNE(n_components=2).fit_transform(Z_test)

    fig, ax = plt.subplots()

    classes = np.unique(Y_test)
    for _class in classes:
        index = Y_test == _class
        ax.plot(test2[index,0], test2[index,1], '.', label = str(_class))#, c=Y_test)

    #plt.legend()
        

    #wandb.log({"test_t_sne": fig})

    plt.savefig('test_t_sne.png')
    plt.close()




if __name__ == '__main__':
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    from TS2VEC import TS2VEC
    model = TS2VEC(input_dim=12, hidden_dim=64, output_dim=320)

    PATH = '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/results/maybe_works/(08_04_2024)_(21_03_13)/best_model.pt'


    #print([test[0] for test in test_])
    H = model.load_state_dict(torch.load(PATH))

    from dataloader import PTB_XL
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
    
    tsne(H, train_dataloader, test_dataloader, DEVICE)
    

