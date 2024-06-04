
# imports
from pathlib import Path
import os
import argparse
import torch
import wandb
import numpy as np
from datetime import datetime

# own functions
from utils import save_parameters, random_seed, TimeTaking, train_test_dataset


# pytorch device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {DEVICE}")


# parameters
seed = 1
dataset_name = 'PTB_XL'
N_train = 3000
N_test = 2000
normalize = False
N_shards = 5
N_slices = 3
hidden_dim = 32
output_dim = 64
classifier = 'logistic'
sensitive_points = 256
n_epochs =[5, 5, 5]
batch_size = 8
learning_rate = 1e-3
p = 0.5
grad_clip = None
alpha = 0.5

N_rep = 11


# paths
data_path = os.path.join(Path.cwd(), 'PTB_XL')

results_path = 'results'
final_path = 'data_pruning'

save_path = os.path.join(results_path, dataset_name, final_path)


# seed
random_seed(1)

# load data
if dataset_name == 'PTB_XL':
    from dataset import PTB_XL
    dataset = PTB_XL(data_path)
else:
    from dataset import AEON_DATA
    # UCR and UEA datasets
    dataset = AEON_DATA(dataset_name)


train_dataset, test_dataset, D = train_test_dataset(dataset=dataset,
                                                    test_proportion=0.3,
                                                    train_size=N_train,
                                                    test_size=N_test,
                                                    seed=seed,
                                                    return_stand=normalize)         



from data_pruning import Pruning

save_accuracy = np.zeros((N_rep, 4)) 
# [before_test_accuracy, before_unlearn_accuracy]
# [after_test_accuracy, after_unlearn_accuracy]

save_mia = np.zeros((N_rep, 4*4))


for rep in range(N_rep):
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

    time = TimeTaking(save_path=save_path)

    data_pruning.load(save_path=save_path , device=DEVICE)



    ################################################
    ###            pre-EVALUATION                ###
    ################################################


    #* Classifier accuracy

    from torch.utils.data import Subset

    unlearning_data = Subset(train_dataset.dataset, 
                             train_dataset.indices[(rep) * sensitive_points: (rep + 1) * sensitive_points])




    ind_train_acc = data_pruning.train_classifiers()


    unlearn_acc, _= data_pruning.evaluate(unlearning_data)


    test_acc, _ = data_pruning.evaluate(test_dataset)

    save_accuracy[rep, 0] = test_acc
    save_accuracy[rep, 1] = unlearn_acc



    #* MIA

    #from dataset import mia_train, mia_data
    from utils import mia_train_test_dataset
    from mia import MIA

    # # train data
    # train_mia = mia_train(train_dataset=train_dataset,
    #                       test_dataset=test_dataset,
    #                       N_train=None,
    #                       N_test=None)


    mia = MIA(device=DEVICE)



    mia_train, mia_test = mia_train_test_dataset(in_dataset=unlearning_data, 
                                                 out_dataset=test_dataset, 
                                                 N_train=None, 
                                                 N_test=len(unlearning_data), 
                                                 seed=1)
    
    print(len(mia_train), len(mia_test))

    train_accuracy = mia.train(model=data_pruning, 
                               train_data=mia_train)
    
    test_accuracy = mia.evaluate(model=data_pruning,
                                 data=mia_test)

    save_mia[rep, :4] = np.array(train_accuracy)
    save_mia[rep, 4:8] = np.array(test_accuracy)


    ##############################################
    ###               UNLEARN                   ##
    ##############################################


    time_ = data_pruning.unlearn(indices=train_dataset.indices[(rep) * sensitive_points: (rep + 1) * sensitive_points],
                                n_epochs=n_epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                grad_clip=grad_clip,
                                alpha=alpha,
                                wandb=wandb,
                                save_path=save_path,
                                time_taking=time,
                                test_dataset=test_dataset)


    ################################################
    ###             post-EVALUATION              ###
    ################################################


    #* Classifier accuracy

    ind_train_acc = data_pruning.train_classifiers()


    unlearn_acc, _ = data_pruning.evaluate(unlearning_data)


    test_acc, _ = data_pruning.evaluate(test_dataset)

    save_accuracy[rep, 2] = test_acc
    save_accuracy[rep, 3] = unlearn_acc


    #* MIA

    mia = MIA(device=DEVICE)


    mia_train, mia_test = mia_train_test_dataset(in_dataset=unlearning_data, 
                                                 out_dataset=test_dataset, 
                                                 N_train=None, 
                                                 N_test=len(unlearning_data), 
                                                 seed=1)
    
    train_accuracy = mia.train(model=data_pruning, 
                               train_data=mia_train)
    
    test_accuracy = mia.evaluate(model=data_pruning,
                                 data=mia_test)
    
    save_mia[rep, 8:12] = np.array(train_accuracy)
    save_mia[rep, 12:16] = np.array(test_accuracy)


    #* Save
    np.save(os.path.join(save_path, 'save_accuracy.npy'), save_accuracy)
    np.save(os.path.join(save_path, 'save_mia.npy'), save_mia)
