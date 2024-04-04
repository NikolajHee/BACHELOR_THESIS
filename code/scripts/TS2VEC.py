"""
General framework for the model described by TS2VEC

"""

import torch
from torch import nn
import warnings
import numpy as np
from code.scripts.classifier import classifier_train
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from code.scripts.utils import baseline


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"device: {DEVICE}")

# TODO: Make faster
# TODO: Learn how the loss works
# TODO: torch-seed the model
# TODO: have fun



class input_projection_layer(nn.Module):
    """
    Projection for each timestamp
        Feature dimension is 12

        input_dim : 
        output_dim : 
        device : 'cpu' or 'cuda'
    """

    def __init__(self, input_dim: int, output_dim: int, device):
        super().__init__()
        self.fully_connected = nn.Linear(in_features=input_dim, 
                                        out_features=output_dim,
                                        bias=True,
                                        device=device)

    def __call__(self, x: torch.tensor):
        return self.fully_connected(x)



class timestamp_masking_layer(nn.Module):
    """
    Randomly masking some of the values from the representation
        only used in training
    """
    def __init__(self, p=0.5, verbose=False):
        super().__init__()
        self.p = p
        self.verbose = verbose


    def __call__(self, r_it):
        """
        r_it : the latent vector
        """
        [N, r, c] = r_it.shape

        masking = torch.rand((N, r, c)) > (1-self.p)

        r_it[masking] = 0
        return r_it



def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


class random_cropping(nn.Module):
    """
    Cropping at random
        works only for 1 dimensional right now?
        Only used in training
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

        self.temporal_unit = 0
    
    def __call__(self, x):
        [N, T, D] = x.shape
       
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)

        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        signal1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        signal2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)

        return signal1[:, -crop_l:], signal2[:, :crop_l]



class CNN_block(nn.Module):
    def __init__(self, l, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, padding='same', kernel_size=3, dilation=2**l)
        self.conv2 = nn.Conv1d(dim, dim, padding='same', kernel_size=3, dilation=2**l)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)

        return x + residual


class dilated_CNN_layer(nn.Module):
    """
    Hmm a bit difficult.

    Ten residual blocks
    """
    def __init__(self, dim):
        super().__init__()
        self.CNN = nn.Sequential(*[
            CNN_block(l=i, dim=dim) 
            for i in range(10)
        ])

    
    def __call__(self, x):
        # input x is (batch_size, T, dimension)
        return self.CNN(x)




class TS2VEC(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=320, # Same as ts2vec 
                 p = 0.5, 
                 device = 'cpu', 
                 verbose=False):
        
        super().__init__()
        self.output_dim = output_dim
        self.input_project_layer = input_projection_layer(input_dim=input_dim,
                                                            output_dim=output_dim,
                                                            device = device)
        
        self.time_masking_layer = timestamp_masking_layer(p=p, verbose=verbose)


        self.dilated_cnn_layer = dilated_CNN_layer(dim=output_dim)



    def forward(self, x):

        r_it = self.input_project_layer(x)

        r_i = self.time_masking_layer(r_it)

        r_i = r_i.transpose(1,2)

        z = self.dilated_cnn_layer(r_i)

        return(z)




def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss



def train(classifier,
          dataset,
          output_dim=256, 
          n_epochs=30, 
          batch_size=10,
          learning_rate=0.001,
          p=0.5,
          input_dim=12,
          grad_clip=0.01,
          verbose=False,
          N_train=1000,
          N_test=100):

    from code.scripts.utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                        test_proportion=0.1,
                                                        train_size=N_train,
                                                        test_size=N_test,
                                                        verbose=verbose,
                                                        seed=0)

    base = baseline(train_dataset=train_dataset,
                    test_dataset=test_dataset)

    model = TS2VEC(input_dim=input_dim, 
                   output_dim=output_dim, 
                   p=p, 
                   device=DEVICE, 
                   verbose=verbose).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss_save = np.zeros((n_epochs))
    test_loss_save = np.zeros((n_epochs))
    train_accuracy_save = np.zeros((n_epochs + 1))
    test_accuracy_save = np.zeros((n_epochs + 1))
    

    for epoch in range(n_epochs):
        # new shuffle for each epoch
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        

        train_accuracy, test_accuracy = classifier_train(classifier, 
                                                         model, 
                                                         train_loader=train_dataloader, 
                                                         test_loader=test_dataloader, 
                                                         device=DEVICE)

        train_accuracy_save[epoch] = train_accuracy
        test_accuracy_save[epoch] = test_accuracy

        print(f"Train accuracy {train_accuracy}. Test accuracy {test_accuracy}. Base {base}.")
        
        train_loss, test_loss = [], []

        for i, (X, Y) in enumerate(train_dataloader):
            X = X.to(DEVICE)

            crop = random_cropping(False)

            signal_aug_1, signal_aug_2 = crop(X)

            optimizer.zero_grad()

            z1 = model.forward(signal_aug_1.float())
            z2 = model.forward(signal_aug_2.float())

            loss = hierarchical_contrastive_loss(z1,  z2)
            if i%1==0: print(f"Epoch: {epoch}. Iter: {i}. HierLoss: {loss}.")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()

            train_loss.append(loss.item())
        

        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(DEVICE)

            crop = random_cropping(False)

            signal_aug_1, signal_aug_2 = crop(X)

            z1 = model.forward(signal_aug_1.float())
            z2 = model.forward(signal_aug_2.float())

            loss = hierarchical_contrastive_loss(z1,  z2)

            test_loss.append(loss.item())

        
        train_loss_save[epoch] = np.mean(train_loss)
        test_loss_save[epoch] = np.mean(test_loss)



    train_accuracy, test_accuracy = classifier_train(classifier, 
                                                        model, 
                                                        train_loader=train_dataloader, 
                                                        test_loader=test_dataloader, 
                                                        device=DEVICE)

    train_accuracy_save[-1] = train_accuracy
    test_accuracy_save[-1] = test_accuracy

    print(f"Train accuracy {train_accuracy}. Test accuracy {test_accuracy}. Base {base}.")
    print('Finished training TS2VEC')

    return train_loss_save, test_loss_save, train_accuracy_save, test_accuracy_save, base








if __name__ == '__main__':
    pass