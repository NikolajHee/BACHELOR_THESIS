"""

"""
import argparse

#* argparse
parser = argparse.ArgumentParser(
                    prog='Main framework of Bachelor Thesis',
                    description='Representation learning and Machine Unlearning',
                    epilog='By Nikolaj Hertz s214644')
parser.add_argument('--dataset', default='PTB_XL') 

args = parser.parse_args()
arguments = vars(args)



# import libraries
import os
import argparse
import torch
from datetime import datetime
import numpy as np

# own functions
from base_framework.utils import save_parameters, random_seed, TimeTaking

# pytorch device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {DEVICE}")


seed = 1
dataset_name = arguments['dataset']
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
results_path = 'results'





#raise NotImplementedError
# save path
save_path = os.path.join(results_path, dataset_name)

# initialize time object
time_object = TimeTaking(save_path=save_path)

# create save path
if not os.path.exists(save_path):
    os.makedirs(save_path)


from pathlib import Path
data_path = os.path.join(Path.cwd(), 'PTB_XL')


# load data
if dataset_name == 'PTB_XL':
    from base_framework.dataset import PTB_XL
    dataset = PTB_XL(data_path)
else:
    from base_framework.dataset import AEON_DATA
    # UCR and UEA datasets
    dataset = AEON_DATA(dataset_name)




N_rep = 20

acc_save = np.zeros(N_rep)
baseline_save = np.zeros(N_rep)
init_save = np.zeros(N_rep)

from base_framework.utils import train_test_dataset
train_dataset, test_dataset, D = train_test_dataset(dataset=dataset,
                                                    test_proportion=0.3,
                                                    train_size=N_train,
                                                    test_size=N_test,
                                                    return_stand=False,
                                                    seed=seed)

for i in range(N_rep):
    # create train/test-spli                                       


    from base_framework.TS2VEC import TS2VEC

    # initialize model
    model = TS2VEC(input_dim=D,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    p=p,
                    device=DEVICE)


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)


    # train the framework
    results, init_ = model.train(train_dataset=train_dataset,
                                test_dataset=test_dataset,
                                optimizer=optimizer,
                                n_epochs=n_epochs,
                                batch_size=batch_size,
                                grad_clip=grad_clip,
                                alpha=alpha,
                                wandb=None,
                                train_path=save_path,
                                t_sne=False,
                                classifier=classifier,
                                time_object=time_object,
                                )
    
    acc_save[i] = results["classifier/test_accuracy"]
    baseline_save[i] = results["classifier/baseline"]
    init_save[i] = init_["classifier/test_accuracy"]

    np.save(os.path.join(save_path, 'ts_test_acc_new.npy'), acc_save)
    np.save(os.path.join(save_path, 'baseline.npy'), baseline_save)
    np.save(os.path.join(save_path, 'init_save.npy'), init_save)

