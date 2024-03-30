"""
General framework for the model described by TS2VEC

"""

import torch
from torch import nn
import warnings
import numpy as np
from torch.distributions.bernoulli import Bernoulli


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

        if self.verbose: 
            print(f"Recieved {r} signals with length {c}.")
            print(f"Masking {masking.sum()} values.")
        r_it[masking] = 0
        return r_it



def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


class random_cropping:
    """
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

class random_cropping2(nn.Module):
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

        np.random.seed(1)
       
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)

        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        #print(crop_offset + crop_eleft, crop_right - crop_eleft)
        #print(crop_offset + crop_left, crop_eright - crop_left)
        #print(crop_l)
        test1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        test2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)



        return ({"s": test1[:, -crop_l:], "left": crop_offset + crop_eleft, "right": crop_right - crop_eleft}, 
                {"s": test2[:, :crop_l], "left": crop_offset + crop_left, "right": crop_right - crop_eleft}, 
                crop_l)



class CNN_block(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv1 = nn.Conv1d(256, 256, padding='same', kernel_size=3, dilation=2**l)
        self.conv2 = nn.Conv1d(256, 256, padding='same', kernel_size=3, dilation=2**l)

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
    def __init__(self, ):
        super().__init__()
        self.CNN = nn.Sequential(*[
            CNN_block(i) 
            for i in range(2)
        ])

    
    def __call__(self, x):
        # input x is (batch_size, T, dimension)
        return self.CNN(x)


# test = input_projection_layer(12, 256, 'cpu')

# x = test(some_signals)

# x = x.transpose(1,2)

# test2 = dilated_CNN_layer()

# print(test2(x))




class TS2VEC(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=320, # Same as ts2vec 
                 p = 0.5, 
                 device = 'cpu', 
                 verbose=False):
        
        super().__init__()
        self.input_project_layer = input_projection_layer(input_dim=input_dim,
                                                            output_dim=output_dim,
                                                            device = device)
        
        self.time_masking_layer = timestamp_masking_layer(p=p, verbose=verbose)


        self.dilated_cnn_layer = dilated_CNN_layer()



    def forward(self, x):

        r_it = self.input_project_layer(x)

        r_i = self.time_masking_layer(r_it)

        r_i = r_i.transpose(1,2)

        z = self.dilated_cnn_layer(r_i)

        return(z)



# test = TS2VEC(12, 256, 0.5, 'cpu', True)

# test_output = test.forward(some_signals)

# print(test_output.shape)


# hierarcical contrasting
import torch
from torch import nn
import torch.nn.functional as F

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



def train(verbose=False, output_dim=256, batches=3, n_epochs=30, batch_size=10):

    from data import PTB_XL

    

    model = TS2VEC(12, output_dim, 0.5, 'cpu', False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #n_iter = 3
    #n_epochs  = 30
    
    loss_save = np.zeros((n_epochs, batches))

    for epoch in range(n_epochs):
        dataset = PTB_XL(batch_size=batch_size, shuffle_=True)
        for i in range(batches):
            X = dataset.load_some_signals()

            crop = random_cropping2(False)

            test1, test2, crop_l = crop(X)

            optimizer.zero_grad()

            z1 = model.forward(test1['s'])
            z2 = model.forward(test2['s'])

            loss = hierarchical_contrastive_loss(z1,  z2)
            if i%10==0 and verbose: print(f"Epoch: {epoch}. Iter: {i}. Loss: {loss}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            loss_save[epoch, i] = loss
            
            optimizer.step()
    print('Finished')
    return loss_save








if __name__ == '__main__':
    import matplotlib.pyplot as plt


    loss_save = train()

    loss_mean = np.mean(loss_save, axis=1)

    plt.plot(loss_mean)
    plt.title('loss over epochs')
    plt.show()