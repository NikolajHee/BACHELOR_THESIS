
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


seed = 1
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
sensitive_points = 256

#from utils import random_seed
#random_seed(seed)

data_path = os.path.join(Path.cwd(), 'PTB_XL')

results_path = 'results'
final_path = 'amnesiac_unlearning'

save_path = os.path.join(results_path, dataset_name, final_path)


from base_framework.utils import random_seed

random_seed(1)

if dataset_name == 'PTB_XL':
    from base_framework.dataset import PTB_XL_v2
    dataset = PTB_XL_v2(data_path)
else:
    from base_framework.dataset import AEON_DATA_v2
    # UCR and UEA datasets
    dataset = AEON_DATA_v2(dataset_name)


from base_framework.utils import train_test_dataset
train_dataset, test_dataset, D = train_test_dataset(dataset=dataset,
                                                    test_proportion=0.3,
                                                    train_size=N_train,
                                                    test_size=N_test,
                                                    seed=seed,
                                                    return_stand=normalize)         

N_rep = 11


import glob

from base_framework.amnesiac import AmnesiacTraining

save_accuracy = np.zeros((N_rep, 4)) 
# [before_test_accuracy, before_unlearn_accuracy]
# [after_test_accuracy, after_unlearn_accuracy]

save_mia = np.zeros((N_rep, 4*4))

extra_mia = np.zeros((N_rep, 4))

for rep in range(N_rep):
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
                                sensitive_points=train_dataset.indices[rep*sensitive_points:(rep+1)*sensitive_points])
    

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




    ################################################
    ###            pre-EVALUATION                ###
    ################################################


    #* Classifier accuracy

    from torch.utils.data import Subset

    unlearning_data = Subset(train_dataset.dataset, 
                             train_dataset.indices[rep*sensitive_points:(rep+1)*sensitive_points])



    train_acc, test_acc, unlearn_acc = model.evaluate(train_dataset=train_dataset,
                                                  test_dataset=test_dataset,
                                                  unlearn_dataset=unlearning_data,
                                                  classifier_name='logistic',
                                                  device=DEVICE)


    save_accuracy[rep, 0] = test_acc
    save_accuracy[rep, 1] = unlearn_acc



    #* MIA
    #from dataset import mia_train, mia_data
    from base_framework.utils import mia_train_test_dataset
    from base_framework.mia import MIA, mia_loss

    # # train data
    # train_mia = mia_train(train_dataset=train_dataset,
    #                       test_dataset=test_dataset,
    #                       N_train=None,
    #                       N_test=None)


    mia = MIA(device=DEVICE)

    mia2 = mia_loss(device=DEVICE)




    mia_train, mia_test = mia_train_test_dataset(in_dataset=unlearning_data, 
                                                 out_dataset=test_dataset, 
                                                 N_train=None, 
                                                 N_test=len(unlearning_data), 
                                                 seed=1)
    

    mia2.train(model, mia_train)

    test1 = mia2.evaluate(model, mia_train)
    test2 = mia2.evaluate(model, mia_test)

    extra_mia[rep, 0] = test1
    extra_mia[rep, 1] = test2
    

    train_accuracy = mia.train(model=model, 
                               train_data=mia_train)
    
    test_accuracy = mia.evaluate(model=model,
                                 data=mia_test)

    save_mia[rep, :4] = np.array(train_accuracy)
    save_mia[rep, 4:8] = np.array(test_accuracy)


    ##############################################
    ###               UNLEARN                   ##
    ##############################################


    time = TimeTaking(save_path=save_path)
    
    
    model.unlearn(sp=train_dataset.indices[rep*sensitive_points:(rep+1)*sensitive_points], 
                                                save_path=save_path, 
                                                time_taking=time)

    
    ################################################
    ###             post-EVALUATION              ###
    ################################################
    

    #* Classifier accuracy

    train_acc, test_acc, unlearn_acc = model.evaluate(train_dataset=train_dataset,
                                                test_dataset=test_dataset,
                                                unlearn_dataset=unlearning_data,
                                                classifier_name='logistic',
                                                device=DEVICE)


    save_accuracy[rep, 2] = test_acc
    save_accuracy[rep, 3] = unlearn_acc


    #* MIA



    mia = MIA(device=DEVICE)

    mia2 = mia_loss(device=DEVICE)


    mia_train, mia_test = mia_train_test_dataset(in_dataset=unlearning_data, 
                                                 out_dataset=test_dataset, 
                                                 N_train=None, 
                                                 N_test=len(unlearning_data), 
                                                 seed=1)
    

    mia2.train(model, mia_train)

    test1 = mia2.evaluate(model, mia_train)
    test2 = mia2.evaluate(model, mia_test)

    extra_mia[rep, 2] = test1
    extra_mia[rep, 3] = test2

    train_accuracy = mia.train(model=model, 
                               train_data=mia_train)
    
    test_accuracy = mia.evaluate(model=model,
                                 data=mia_test)
    
    save_mia[rep, 8:12] = np.array(train_accuracy)
    save_mia[rep, 12:16] = np.array(test_accuracy)



    #* Save
    np.save(os.path.join(save_path, 'save_accuracy_new.npy'), save_accuracy)
    np.save(os.path.join(save_path, 'save_mia_new.npy'), save_mia)
    np.save(os.path.join(save_path, 'extra_mia_new.npy'), extra_mia)
