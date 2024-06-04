
from pathlib import Path

# import libraries
import os
import argparse
import torch
import wandb
from datetime import datetime
import numpy as np

# own functions
from utils import save_parameters, random_seed, TimeTaking


# pytorch device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {DEVICE}")


seed = 1
dataset_name = 'PTB_XL'
N_train = 3000
N_test = 2000
normalize =False
N_shards = 5
N_slices = 3
hidden_dim = 32
output_dim = 64
classifier = 'logistic'
n_epochs =[5, 5, 5]
batch_size = 8
learning_rate = 1e-3
p = 0.5
grad_clip = None
alpha = 0.5

#from utils import random_seed
#random_seed(seed)

data_path = os.path.join(Path.cwd(), 'PTB_XL')

results_path = 'results'
final_path = 'data_pruning'

save_path = os.path.join(results_path, dataset_name, final_path)


from utils import random_seed

random_seed(1)

if dataset_name == 'PTB_XL':
    from dataset import PTB_XL
    dataset = PTB_XL(data_path)
else:
    from dataset import AEON_DATA
    # UCR and UEA datasets
    dataset = AEON_DATA(dataset_name)


from utils import train_test_dataset
train_dataset, test_dataset, D = train_test_dataset(dataset=dataset,
                                                    test_proportion=0.3,
                                                    train_size=N_train,
                                                    test_size=N_test,
                                                    seed=seed,
                                                    return_stand=normalize)         
sensitive_points = [2, 4, 8, 16, 32, 64, 128]

N_rep = 20

training_time = np.zeros((N_rep, len(sensitive_points)))

from data_pruning import Pruning

for i in range(13, N_rep):
    for j, sp in enumerate(sensitive_points):
        print('-'*20)
        print(f"Repetition: ({i}). Amount of sensitive points: ({sp}).")
        print('-'*20)

        data_pruning = Pruning(dataset=train_dataset, 
                                N_shards=N_shards, 
                                N_slices=N_slices, 
                                input_dim=D, 
                                hidden_dim=hidden_dim, 
                                output_dim=output_dim, 
                                p=p, 
                                device=DEVICE,
                                classifier_name=classifier,
                                seed=seed)

        data_pruning.load(save_path=save_path , device=DEVICE)

        time = TimeTaking(save_path=save_path)

        start_indice = 0 + i*128
        end_indice = sp + i*128


        time_ = data_pruning.unlearn(indices=train_dataset.indices[start_indice:end_indice],
                                    n_epochs=n_epochs,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    grad_clip=grad_clip,
                                    alpha=alpha,
                                    wandb=None,
                                    save_path=save_path,
                                    time_taking=time,
                                    test_dataset=test_dataset)
        
        training_time[i,j] = time_['Overall Unlearning']

        np.save(os.path.join(save_path, 'training_time.npy'), training_time)
        
