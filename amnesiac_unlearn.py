

# python unlearn.py -sp 10 -a --seed 0 -hd 32 -od 64 -ne 5 --N_train 200 --N_test 100

from pathlib import Path

# import libraries
import os
import argparse
import torch
import pickle
import wandb
from datetime import datetime

# own functions
from utils import save_parameters, random_seed, TimeTaking


# pytorch device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {DEVICE}")


seed = 1
N_train = None
N_test = None
normalize = False
hidden_dim = 64
output_dim = 320
sensitive_points = 25
p = 0.5
n_epochs = [20]
batch_size = 8
learning_rate = 1e-3
grad_clip = None
alpha = 0.5
classigitfier = 'logsitic'


from utils import random_seed
random_seed(seed)

data_path = os.path.join(Path.cwd(), 'PTB_XL')

results_path = 'results'
final_path = 'amnesiac_unlearning'
dataset_name = 'PTB_XL'

save_path = os.path.join(results_path, dataset_name, final_path)



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
from amnesiac import AmnesiacTraining
model = AmnesiacTraining(input_dim=D,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            p=p,
                            device=DEVICE,
                            sensitive_points=train_dataset.indices[0:sensitive_points])


model.load(os.path.join(save_path, 'model.pt'))



from torch.utils.data import Subset

unlearn_dataset = Subset(train_dataset.dataset, train_dataset.indices[0:sensitive_points])


train_acc, test_acc, unlearn_acc = model.evaluate(train_dataset=train_dataset,
                                                  test_dataset=test_dataset,
                                                  unlearn_dataset=unlearn_dataset,
                                                  classifier_name='logistic',
                                                  device=DEVICE)

save = {"train_acc": train_acc,
        "unlearn_acc": unlearn_acc,
        "test_acc": test_acc}


with open(dataset_name + '_' + 'amnesiac' + '_before_accuracy.pickle', 'wb') as handle:
    pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)


time = TimeTaking(save_path=save_path)

model.unlearn(save_path, time_taking=time)



train_acc, test_acc, unlearn_acc = model.evaluate(train_dataset=train_dataset,
                                                  test_dataset=test_dataset,
                                                  unlearn_dataset=unlearn_dataset,
                                                  classifier_name='logistic',
                                                  device=DEVICE)


save = {"train_acc": train_acc,
        "unlearn_acc": unlearn_acc,
        "test_acc": test_acc}



with open(dataset_name + '_' + 'amnesiac' + '_after_accuracy.pickle', 'wb') as handle:
    pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)



