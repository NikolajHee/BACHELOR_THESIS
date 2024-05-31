
from pathlib import Path

# import libraries
import os
import argparse
import torch
import wandb
from datetime import datetime

# own functions
from utils import save_parameters, random_seed, TimeTaking


# pytorch device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {DEVICE}")




seed = 1
dataset_name = 'PTB_XL'
N_train = None
N_test = None
normalize = False
N_shards = 10
N_slices = 5
hidden_dim = 64
output_dim = 320
classifier = 'logistic'
sensitive_points = 25
n_epochs =[5, 5, 5, 5, 5]
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





from data_pruning import Pruning

data_pruning = Pruning(dataset=train_dataset, 
                    N_shards=N_shards, 
                    N_slices=N_slices, 
                    input_dim=D, 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    p=p, 
                    device=DEVICE,
                    classifier_name=classifier)

time = TimeTaking(save_path=save_path)

data_pruning.load(save_path=save_path , device=DEVICE)


from torch.utils.data import Subset

print('training classifiers')

ind_train_acc = data_pruning.train_classifiers()

print(ind_train_acc)
print('unlearn acc')

unlearn_acc = data_pruning.evaluate(Subset(train_dataset.dataset, train_dataset.indices[0:sensitive_points]))

print(unlearn_acc)
print('test acc')
test_acc = data_pruning.evaluate(test_dataset)

print(test_acc)

save = {'ind_train_acc': ind_train_acc,
        "unlearn_acc": unlearn_acc,
        "test_acc": test_acc}


import pickle

with open('before_accuracy.pickle', 'wb') as handle:
    pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)




time_ = data_pruning.unlearn(indices=train_dataset.indices[0:sensitive_points],
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            grad_clip=grad_clip,
                            alpha=alpha,
                            wandb=wandb,
                            save_path=save_path,
                            time_taking=time,
                            test_dataset=test_dataset)

print(time_)


print(ind_train_acc)

unlearn_acc = data_pruning.evaluate(data_pruning.data.unlearned_points)

print(unlearn_acc)

test_acc = data_pruning.evaluate(test_dataset)

print(test_acc)

save = {'ind_train_acc': ind_train_acc,
        "unlearn_acc": unlearn_acc,
        "test_acc": test_acc}



with open('after_accuracy.pickle', 'wb') as handle:
    pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)

