"""
General framework for the model described by TS2VEC

"""

# imports
import torch
from torch import nn
import warnings
import numpy as np
from classifier import classifier_train
from clustering import tsne
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import baseline
import os
from tqdm import tqdm


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"device: {DEVICE}")




class InputProjectionLayer(nn.Module):
    """
    Projection for each timestamp
        Feature dimension is 12

        input_dim : input dimension of data
        output_dim : the latent dimension
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



class TimestampMaskingLayer(nn.Module):
    """
    Randomly masking some of the values from the representation
        only used in training
    """

    def __init__(self):
        super().__init__()


    def __call__(self, r_it, p):
        """
        r_it : the latent vector
        """
        [N, r, c] = r_it.shape

        masking = torch.rand((N, r, c)) > (1-p)

        r_it[masking] = 0
        return r_it



def take_per_row(A, indx, num_elem):
    """
    From github of ts2vec
    """
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


class RandomCropping(nn.Module):
    """
    Cropping at random
        Only used in training

    """

    def __init__(self):
        super().__init__()
        self.temporal_unit = 0
    
    def __call__(self, x):

        # TOOO: how does this work?
       
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)

        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        signal1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        signal2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)

        return signal1, signal2, crop_l



class CNN_block(nn.Module):
    """
    dumped
    """
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
    dumped
    """
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


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)


class TS2VEC(nn.Module):
    """
    Main model class. 

    Consists of three layers
        InputProjectionLayer
        TimestampMaskingLayer
        DilatedConvEncoder

    input_dim : feature dimension D of the time series
    output_dim : latent dimension for each timestamp
    hidden_dim : dimension of the convolutional layers
    p : amount of masking applied during training
    device : device used during training
    verbose : verbose
    """

    def __init__(self, 
                 input_dim, 
                 output_dim=320, # Same as ts2vec 
                 hidden_dim=64,
                 p = 0.5, 
                 device = 'cpu', 
                 verbose=False):
        
        super().__init__()
        self.output_dim = output_dim
        self.test = False  # variable for test/train-mode
        self.p = p


        # initialising layers

        self.input_project_layer = InputProjectionLayer(input_dim=input_dim,
                                                            output_dim=hidden_dim,
                                                            device = device)
        
        self.time_masking_layer = TimestampMaskingLayer()

        #self.dilated_cnn_layer = dilated_CNN_layer(dim=output_dim)

        # WARNING: hardcoded to be 10 blocks
        self.dilated_cnn_layer = DilatedConvEncoder(in_channels=hidden_dim, channels=[hidden_dim]*10 + [output_dim], kernel_size=3)

            


    def forward(self, x):

        r_it = self.input_project_layer(x)

        # no instances should be masked, if it is test
        r_i = r_it if self.test else self.time_masking_layer(r_it, p=self.p)

        r_i = r_i.transpose(1,2)

        z = self.dilated_cnn_layer(r_i)

        return z




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
          hidden_dim=64,
          output_dim=256, 
          n_epochs=30, 
          batch_size=10,
          learning_rate=0.001,
          p=0.5,
          input_dim=12,
          grad_clip=None,
          verbose=False,
          N_train=1000,
          N_test=100,
          wandb=None,
          train_path=None,
          classify=False):
    """
    Main training function
    """
    
    best_test_error = np.inf

    from utils import train_test_dataset
    train_dataset, test_dataset, mean, std = train_test_dataset(dataset=dataset,
                                                        test_proportion=0.3,
                                                        train_size=N_train,
                                                        test_size=N_test,
                                                        verbose=verbose,
                                                        seed=0,
                                                        return_stand=True)

    #print(len(train_dataset), len(test_dataset))

    base = baseline(train_dataset=train_dataset,
                    test_dataset=test_dataset)

    model_ = TS2VEC(input_dim=input_dim, 
                   output_dim=output_dim, 
                   hidden_dim=hidden_dim,
                   p=p, 
                   device=DEVICE, 
                   verbose=verbose).to(DEVICE)
    
    model = torch.optim.swa_utils.AveragedModel(model_)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_loss_save = np.zeros((n_epochs))
    test_loss_save = np.zeros((n_epochs))
    train_accuracy_save = np.zeros((n_epochs + 1))
    test_accuracy_save = np.zeros((n_epochs + 1))
    

    for epoch in range(n_epochs):
        # new shuffle for each epoch
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        model.test = False

        train_loss_list, test_loss_list = [], []
        print(f"Epoch: {epoch}")

        for i, (X, Y) in tqdm(enumerate(train_dataloader)):
            
            X = X.to(DEVICE)
            X = (X - mean)/std

            crop = RandomCropping()

            #print(X.shape, type(X), X)
            signal_aug_1, signal_aug_2, crop_l = crop(X)
            
            optimizer.zero_grad()

            z1 = model.forward(signal_aug_1.float())
            z2 = model.forward(signal_aug_2.float())

            z1 = z1.reshape(batch_size, -1, output_dim)
            z2 = z2.reshape(batch_size, -1, output_dim)

            train_loss = hierarchical_contrastive_loss(z1[:, -crop_l:],  z2[:,:crop_l])


            #if i%20==0: print(f"Train loss: {train_loss}.")

            train_loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            model.update_parameters(model_)

            train_loss_list.append(train_loss.item())
        
        model.test = True

        for i, (X, y) in tqdm(enumerate(test_dataloader)):
            X = X.to(DEVICE)

            X = (X - mean)/std

            crop = RandomCropping()

            signal_aug_1, signal_aug_2, crop_l = crop(X)

            z1 = model.forward(signal_aug_1.float())
            z2 = model.forward(signal_aug_2.float())

            z1 = z1.reshape(batch_size, -1, output_dim)
            z2 = z2.reshape(batch_size, -1, output_dim)

            test_loss = hierarchical_contrastive_loss(z1[:, -crop_l:],  z2[:,:crop_l])

            #if i%1==0: print(f"Test loss: {test_loss}.")

            test_loss_list.append(test_loss.item())

            
        print(f"Epoch {epoch}. Train loss {np.mean(train_loss_list)}. Test loss {np.mean(test_loss_list)}.")
        

        model.test = True

        if classify:
            train_accuracy, test_accuracy = classifier_train(classifier, 
                                                            model, 
                                                            train_loader=train_dataloader, 
                                                            test_loader=test_dataloader, 
                                                            device=DEVICE)
            
        
        t_sne = False
        if t_sne:
            tsne(model,
                 train_loader=train_dataloader,
                 test_loader=test_dataloader,
                 device=DEVICE,
                 wandb=wandb)

        if classify:
            train_accuracy_save[epoch] = train_accuracy
            test_accuracy_save[epoch] = test_accuracy
            print(f"Train accuracy {train_accuracy}. Test accuracy {test_accuracy}. Base {base}.")


        train_loss_save[epoch] = np.mean(train_loss_list)
        test_loss_save[epoch] = np.mean(test_loss_list)

        if wandb is not None:
            wandb.log({"tsloss/train_loss": train_loss, "tsloss/test_loss": test_loss})
            if classify:
                wandb.log({"accuracy/train_accuracy": train_accuracy, "accuracy/test_accuracy": test_accuracy})


        # save the actual model
        torch.save(model.state_dict(), os.path.join(train_path, 'model.pt'))


        # save the best model so far 
        mean_test =  np.mean(test_loss_list)
        if best_test_error > mean_test:
            best_test_error = mean_test
            torch.save(model.state_dict(), os.path.join(train_path, 'best_model.pt'))


    print('Finished training TS2VEC')
    

    return train_loss_save, test_loss_save, train_accuracy_save, test_accuracy_save, base








if __name__ == '__main__':
    from dataloader import PTB_XL
    dataset = PTB_XL()

    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                        test_proportion=0.1,
                                                        verbose=True,
                                                        seed=0,
                                                        return_stand=True)
    

    crop = RandomCropping(False)

    signal_aug_1, signal_aug_2, crop_l = crop(torch.from_numpy(train_dataset[0][0]).reshape(1, 1000, 12))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(signal_aug_1.view(-1, 12).detach())
    ax[1].plot(signal_aug_2.view(-1, 12).detach())
    plt.show()

    model = TS2VEC(input_dim=12, 
                   output_dim=16, 
                   p=0.5, 
                   device=DEVICE, 
                   verbose=True).to(DEVICE)
    
    

    train()

    z1 = model.forward(signal_aug_1.float())
    z2 = model.forward(signal_aug_2.float())




    z1 = z1.reshape(-1, 1000, 16)
    z2 = z2.reshape(-1, 1000, 16)

    test_loss = hierarchical_contrastive_loss(z1,  z2)