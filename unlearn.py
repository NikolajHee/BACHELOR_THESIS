
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


#* argparse
parser = argparse.ArgumentParser(
                    prog='Main framework of Bachelor Thesis',
                    description='Representation learning and Machine Unlearning',
                    epilog='By Nikolaj Hertz s214644')

unlearn = parser.add_mutually_exclusive_group()

unlearn.add_argument('-dp', '--data_pruning', action='store_const', dest='strategy', const='data_pruning')
unlearn.add_argument('-a', '--amnesiac_unlearning', action='store_const', dest='strategy', const='amnesiac_unlearning')


# data pruning specific
parser.add_argument('--N_shards', type=int)
parser.add_argument('--N_slices', type=int)


parser.add_argument('--dataset', default='PTB_XL') 
parser.add_argument('-id', '--input_dim', default=12, type=int)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-n', '--normalize', action='store_true')
parser.add_argument('--seed', default=None, type=int)
#parser.add_argument('--model_path', default=None, nargs='+')


# variables that can be sweeped
parser.add_argument('-c', '--classifier', default=None, choices=['logistic', 'svc'])
parser.add_argument('-hd', '--hidden_dim', default=64, type=int)
parser.add_argument('-od', '--output_dim', default=320, type=int)
parser.add_argument('-bs', '--batch_size', default=8, type=int)
parser.add_argument('-ne', '--n_epochs', default=200, type=int, nargs='+')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('-p', default=0.5, type=float)
parser.add_argument('-gc', '--grad_clip', default=None, type=float)
parser.add_argument('--N_train', default=None, type=int)
parser.add_argument('--N_test', default=None, type=int)
parser.add_argument('--alpha', default=0.5, type=float)


# get arguments from parser
args = parser.parse_args()
arguments = vars(args)

def folder_structure(save_path, N_shards, N_slices):
    save_path = os.path.join(save_path, 'model_save')
    for i in range(N_shards):
        for j in range(N_slices):

            os.makedirs(os.path.join(save_path, f"shard_{i}/slice_{j}"), exist_ok=True)
    
    return save_path


now = datetime.now()
dt_string = now.strftime("(%d_%m_%Y)_(%H_%M_%S)")

wandb.init(project="BACHELOR_THESIS",
            name='UNLEARNING',
            config=arguments)

if args.seed:
    from utils import random_seed
    random_seed(args.seed)

from dataset import PTB_XL
dataset = PTB_XL()

from utils import train_test_dataset
train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                    test_proportion=0.3,
                                                    train_size=args.N_train,
                                                    test_size=args.N_test,
                                                    return_stand=args.normalize)
    


results_path = 'results'

final_path = 'unlearn'

save_path = os.path.join(results_path, args.dataset, final_path)


if args.strategy == 'data_pruning':
    
    from data_pruning import Pruning


    print(f"1.7MB per model. Therefore for {args.N_shards*args.N_slices} models, it needs {args.N_shards*args.N_slices*1.6628:.2f} MB")

    assert args.N_shards*args.N_slices*1.6628 < 1000, \
            f"The model will use above 1000MB in space! " \
            f"It will use around {args.N_shards*args.N_slices*1.6628} MB."
    

    save_path = folder_structure(save_path=save_path, N_shards=args.N_shards, N_slices=args.N_slices)

    data_pruning = Pruning(dataset=train_dataset, 
                     N_shards=args.N_shards, 
                     N_slices=args.N_slices, 
                     input_dim=args.input_dim, 
                     hidden_dim=args.hidden_dim, 
                     output_dim=args.output_dim, 
                     p=args.p, 
                     device=DEVICE,
                     classifier_name=args.classifier)
    
    time = TimeTaking(save_path=save_path)
    
    acc = data_pruning.train(n_epochs=args.n_epochs, 
                             batch_size=args.batch_size, 
                             learning_rate=args.learning_rate, 
                             grad_clip=args.grad_clip, 
                             alpha=args.alpha, 
                             wandb=wandb, 
                             save_path=save_path, 
                             time_taking=time,
                             test_dataset=test_dataset)
    
    print('-'*20)
    print("UNLEARNING")
    print('-'*20)

    print(f'Accuracy: {acc}')

    data_pruning.unlearn(indices=train_dataset.indices[0:3],
                         n_epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         learning_rate=args.learning_rate,
                         grad_clip=args.grad_clip,
                         alpha=args.alpha,
                         wandb=wandb,
                         save_path=save_path,
                         time_taking=time,
                         test_dataset=test_dataset
                )




