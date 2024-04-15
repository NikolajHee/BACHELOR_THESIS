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
import os
from tqdm import tqdm


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

        return signal1, signal2, crop_l, {'l1': crop_offset + crop_eleft, 'r1': crop_offset + crop_eleft + crop_right - crop_eleft,
                                          'l2': crop_offset + crop_left, 'r2': crop_offset + crop_left + crop_eright - crop_left}




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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p):
        """
        The main Encoder of the TS2VEC inspired by TS2VEC.

        Consists of three layers
            InputProjectionLayer
            TimestampMaskingLayer
            DilatedConvEncoder

        inputs: 
                input_dim : feature dimension D of the time series
                hidden_dim : dimension of the output of InputProjectionLayer and main kernels of DilatedConvEncoder.
                output_dim : latent dimension for each timestamp
                p : amount of masking applied during training
                device : device used during training
        """
        super().__init__()
        # Input Projection Layer (outputs feature for each timestamp)
        self.p = p
        self.output_dim = output_dim

        # initialising layers
        self.input_project_layer = nn.Linear(in_features=input_dim, 
                                             out_features=hidden_dim,
                                             bias=True)
        
        # WARNING: hardcoded to be 10 blocks
        self.dilated_cnn_layer = DilatedConvEncoder(in_channels=hidden_dim, 
                                                    channels=[hidden_dim]*10 + [output_dim], 
                                                    kernel_size=3)
        # variable for test/train mode
        self.test = False


    def TimestampMaskingLayer(self, r_it, p):
        """
        Randomly masking some of the values from the representation
            only used in training
        """
        # N is batch_size, T is length of signal, d is dimension
        [N, T, d] = r_it.shape

        # masking:
        masking = torch.rand((N, T)) > (1-p)

        r_it[masking] = 0
        return r_it
    

    def forward(self, x):
        """
        input 
            x : should be of dimension (N, T, D), that is (batch_size, signal length, dimension)
        """
        r_it = self.input_project_layer(x) # N x T x Dr
        # no instances should be masked, if it is in test-mode
        r_i = r_it if self.test else self.TimestampMaskingLayer(r_it, p=self.p) # N x T x Dr
        r_i = r_i.transpose(1,2) # N x Dr x T
        z = self.dilated_cnn_layer(r_i) # N x Do x T
        z = z.transpose(1,2) # N x T x Do
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



class TS2VEC(nn.Module):
    """
    Main model class. 
    """

    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 output_dim,
                 p, 
                 device):
        
        super().__init__()

        self.output_dim = output_dim
        self.device = device

        # init encoder
        self.encoder = Encoder(input_dim=input_dim,
                               hidden_dim=hidden_dim,
                               output_dim=output_dim,
                               p=p).to(device)

        # see article [insert cite]
        self.model = torch.optim.swa_utils.AveragedModel(self.encoder)

        # TODO: why is this nessacary?
        self.model.update_parameters(self.encoder)


    def train(self,
          train_dataset,
          test_dataset,
          n_epochs, 
          batch_size,
          learning_rate,
          grad_clip,
          wandb,
          train_path,
          t_sne,
          classifier):


        # initializing the best seen test error
        best_train_error = np.inf

        # same optimizer as ts2vec; experiment with other choices
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # test the representations by clustering (with self-organizing maps)
        if t_sne:
            print("Clustering...")
            fig_train, fig_test = tsne(self.model,
                                       train_dataloader=train_dataloader,
                                       test_dataloader=test_dataloader,
                                       device=self.device)
            
            wandb.log({"t_sne/train_t_sne": fig_train, "t_sne/test_t_sne": fig_test})

        # test the representations by classifying the labels
        if classifier:
            print("Classifying...")
            train_accuracy, test_accuracy = classifier_train(classifier, 
                                                             self.model, 
                                                             train_loader=train_dataloader, 
                                                             test_loader=test_dataloader, 
                                                             device=self.device)

            wandb.log({"classifier/train_accuracy": train_accuracy, "classifier/test_accuracy": test_accuracy})

        # main training loop
        for epoch in range(n_epochs):
            # new shuffle for each epoch
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # ensure training mode
            self.model.module.test, self.encoder.test = False, False

            train_loss_list, test_loss_list = [], []

            # evaluating each batch
            for i, (X, Y) in tqdm(enumerate(train_dataloader)):
                
                X = X.to(self.device) # N x T x D
                
                # random crop two overlapping augments:
                crop = RandomCropping()

                #print(X.shape)
                
                signal_aug_1, signal_aug_2, crop_l, _ = crop(X) # (N x T1 x D) & (N x T2 & D)
                
                # reset gradients
                optimizer.zero_grad()

                # input augments to model
                z1 = self.model.forward(signal_aug_1.float()) # (N x T1 x Dr)
                z2 = self.model.forward(signal_aug_2.float()) # (N x T2 x Dr)
                

                # evaluate the loss on the overlapping part
                # TODO: Why only on the overlapping part?
                train_loss = hierarchical_contrastive_loss(z1[:, -crop_l:],  z2[:,:crop_l])


                # calculate gradients
                train_loss.backward()
                
                # clip gradient
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                # take gradient step
                optimizer.step()

                # TODO: What does this do?
                self.model.update_parameters(self.encoder)

                train_loss_list.append(train_loss.item())
            
            print('-'*20)
            print(f"Epoch {epoch}. Train loss {np.mean(train_loss_list)}.")
            print('-'*20)
            
            # ensure model test-mode
            self.model.module.test, self.encoder.test = False, False


            # TODO: Fix below
            # log the loss in WandB
            wandb.log({"tsloss/hierarchical_contrastive_loss": np.mean(train_loss_list)})

            # test the representations by classifying the labels
            if classifier and ((epoch%10==0) or (epoch == n_epochs-1)):
                print("Classifying...")
                train_accuracy, test_accuracy = classifier_train(classifier, 
                                                                self.model, 
                                                                train_loader=train_dataloader, 
                                                                test_loader=test_dataloader, 
                                                                device=self.device)

                wandb.log({"classifier/train_accuracy": train_accuracy, "classifier/test_accuracy": test_accuracy})
                



            

            # test the representations by clustering (with self-organizing maps)
            if t_sne and ((epoch%10==0) or (epoch == n_epochs-1)):
                print("Clustering...")
                fig_train, fig_test = tsne(self.model,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    device=self.device)
                
                wandb.log({"t_sne/train_t_sne": fig_train, "t_sne/test_t_sne": fig_test})



            # save the actual model
            torch.save(self.model.state_dict(), os.path.join(train_path, 'model.pt'))


            # save the model with some patience
            # # TODO: consider having some patience or/and threshold (either percentage or absolute)
            mean_train =  np.mean(train_loss_list)
            if best_train_error > mean_train:
                print(f"best model from epoch {epoch + 1}")
                best_train_error = mean_train
                torch.save(self.model.state_dict(), os.path.join(train_path, 'best_model.pt'))


        print('Finished training TS2VEC')











if __name__ == '__main__':
    # load data
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"device: {DEVICE}")

    #from dataset import PTB_XL
    #dataset = PTB_XL()

    from dataset import AEON_DATA
    dataset = AEON_DATA('ElectricDevices')

    # create train/test-split
    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                     test_proportion=0.3,
                                                     train_size=10,
                                                     test_size=10,
                                                     seed=0,
                                                     return_stand=False)
                                                    
    # Either train a model or load existing model

    from TS2VEC import TS2VEC

    model = TS2VEC(input_dim=1,
                    hidden_dim=64,
                    output_dim=320,
                    p=0.5,
                    device=DEVICE)

    model.train(train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_epochs=10,
                batch_size=5,
                learning_rate=0.001,
                grad_clip=None,
                wandb=None,
                train_path=None,
                t_sne=False,
                classifier='logistic',)