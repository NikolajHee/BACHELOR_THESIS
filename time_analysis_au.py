
from pathlib import Path

# import libraries
import os
import argparse
import torch
#import wandb
import numpy as np
from datetime import datetime

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
hidden_dim = 32
output_dim = 64
classifier = 'logistic'
n_epochs = 15
batch_size = 8
learning_rate = 1e-3
p = 0.5
grad_clip = None
alpha = 0.5

#from utils import random_seed
#random_seed(seed)

data_path = os.path.join(Path.cwd(), 'PTB_XL')

results_path = 'results'
final_path = 'amnesiac_unlearning'

save_path = os.path.join(results_path, dataset_name, final_path)


from utils import random_seed

random_seed(1)

if dataset_name == 'PTB_XL':
    from dataset import PTB_XL_v2
    dataset = PTB_XL_v2(data_path)
else:
    from dataset import AEON_DATA_v2
    # UCR and UEA datasets
    dataset = AEON_DATA_v2(dataset_name)


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

import glob

from amnesiac import AmnesiacTraining

for i in range(N_rep):

    if os.path.isdir(save_path):
        items = glob.glob(save_path + '/*.pkl')
        for item in items:
            os.remove(item)
        
    os.makedirs(save_path, exist_ok=True)

    model = AmnesiacTraining(input_dim=D,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                p=p,
                                device=DEVICE,
                                sensitive_points=train_dataset.indices[i*128:(i+1)*128])
    

    model.train(train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            alpha=alpha,
            wandb=None,
            train_path=save_path,
            t_sne=False,
            classifier=classifier)


    for j, sp in enumerate(sensitive_points):
        
        model.load(save_path)

        print('-'*20)
        print(f"amount of sensitive points: {sp}")
        print('-'*20)

        time = TimeTaking(save_path=save_path)
        
        start_indice = 0 + i*128
        end_indice = sp + i*128
        
        training_time[i,j] = model.unlearn(train_dataset.indices[start_indice:end_indice], save_path, time_taking=time)['Overall unlearning']

        np.save(os.path.join(save_path, 'training_time.npy'), training_time)
       
        
