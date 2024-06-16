
from pathlib import Path

# import libraries
import os
import argparse
import torch
#import wandb
import numpy as np
from datetime import datetime

# own functions
from base_framework.utils import save_parameters, random_seed, TimeTaking


# pytorch device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {DEVICE}")


seed = 2
dataset_name = 'PTB_XL'
N_train = 3000
N_test = 2000
normalize = False
hidden_dim = 32
output_dim = 64
classifier = 'logistic'
n_epochs = 10
batch_size = 8
learning_rate = 1e-3
p = 0.5
grad_clip = None
alpha = 0.5

#from utils import random_seed
#random_seed(seed)

data_path = os.path.join(Path.cwd(), 'PTB_XL')

results_path = 'results'

save_path = os.path.join(results_path, dataset_name)


from base_framework.utils import random_seed

random_seed(1)

if dataset_name == 'PTB_XL':
    from base_framework.dataset import PTB_XL
    dataset = PTB_XL(data_path)
else:
    from base_framework.dataset import AEON_DATA_v2
    # UCR and UEA datasets
    dataset = AEON_DATA_v2(dataset_name)


from base_framework.utils import train_test_dataset
    
sensitive_points = [2, 4, 8, 16, 32, 64, 128]

N_rep = 20

training_time = np.zeros((N_rep, len(sensitive_points)))


from base_framework.TS2VEC import TS2VEC

model = TS2VEC(12, 32, 64, 0.5, DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

for i in range(N_rep):

    for j, sp in enumerate(sensitive_points):
        train_dataset, test_dataset, D = train_test_dataset(dataset=dataset,
                                                    test_proportion=0.3,
                                                    train_size=N_train-sp,
                                                    test_size=N_test,
                                                    seed=seed,
                                                    return_stand=normalize)     

        print('-'*20)
        print(f"amount of sensitive points: {sp}")
        print('-'*20)

        time = TimeTaking(save_path=save_path)
        
        training_time[i,j] = model.train(train_dataset=train_dataset,
                                         test_dataset=test_dataset,
                                         optimizer=optimizer,
                                         n_epochs=n_epochs,
                                        batch_size=batch_size,
                                        grad_clip=grad_clip,
                                        alpha=alpha,
                                        wandb=None,
                                        train_path=save_path,
                                        t_sne=False,
                                        classifier=False,
                                        time_object=time,
                                         )['Model Training']

        np.save(os.path.join(save_path, 'ts_training_time.npy'), training_time)
       
        
