"""
Main script of the Bachelor Thesis. 

TODO: Insert Seed

The script can be executed in two ways:

1. As a sweep:
    - The script will execute the main function with all possible combinations of the arguments.
    - The sweep will be saved on wandb.
    - The sweep will be executed on wandb.

2. As a single value training:
    - The script will execute the main functi<on with the given arguments.
    - The results will be saved in a folder named after the current date and time.
"""



# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import torch
import wandb
from utils import save_parameters, remove_list, random_seed

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"device: {DEVICE}")


parser = argparse.ArgumentParser(
                    prog='Main framework of Bachelor Thesis',
                    description='Representation learning and Machine Unlearning',
                    epilog='By Nikolaj Hertz s214644')
parser.add_argument('--dataset', default='PTB_XL') 
parser.add_argument('-id', '--input_dim', default=12, type=int)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-n', '--normalize', action='store_true')
parser.add_argument('--seed', default=None)
parser.add_argument('--t-sne', action='store_true')
parser.add_argument('--model_path', default=None, nargs='+')


# variables that can be sweeped
parser.add_argument('-c', '--classifier', default=[None], choices=['logistic', 'svc'], nargs='+')
parser.add_argument('-hd', '--hidden_dim', default=[64], type=int, nargs='+')
parser.add_argument('-od', '--output_dim', default=[320], type=int, nargs='+')
parser.add_argument('-bs', '--batch_size', default=[8], type=int, nargs='+')
parser.add_argument('-ne', '--n_epochs', default=[200], type=int, nargs='+')
parser.add_argument('-lr', '--learning_rate', default=[0.001], type=float, nargs='+')
parser.add_argument('-p', default=[0.5], type=float, nargs='+')
parser.add_argument('-gc', '--grad_clip', default=[None], type=float, nargs='+')
parser.add_argument('--N_train', default=[None], type=int, nargs='+')
parser.add_argument('--N_test', default=[None], type=int, nargs='+')




args = parser.parse_args()

arguments = vars(args)




#* train function ; can be executed both by a sweep, or a by single values
def main(sweep=True):
    if sweep:
        results_path = 'results/sweep'

        now = datetime.now()
        dt_string = now.strftime("(%d_%m_%Y)_(%H_%M_%S)")

        wandb.init(project="BACHELOR_THESIS",
                   name=dt_string)

    else:
        results_path = 'results'

        now = datetime.now()
        dt_string = now.strftime("(%d_%m_%Y)_(%H_%M_%S)")

        wandb.init(project="BACHELOR_THESIS",
                   name=dt_string,
                   config=arguments)


    config = wandb.config
    save_path = os.path.join(results_path, config.dataset, dt_string)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_parameters(save_path, vars(args))


    # load data
    if config.dataset == 'PTB_XL':
        from dataset import PTB_XL
        dataset = PTB_XL()
    else:
        from dataset import AEON_DATA
        dataset = AEON_DATA(config.dataset)

    print(f"using the data {config.dataset}")

    # create train/test-split
    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                     test_proportion=0.3,
                                                     train_size=config.N_train,
                                                     test_size=config.N_test,
                                                     verbose=config.verbose,
                                                     seed=config.seed,
                                                     return_stand=config.normalize)
                                                    
    # Either train a model or load existing model
    if not args.model_path:
        from TS2VEC import TS2VEC

        model = TS2VEC(input_dim=config.input_dim,
                       hidden_dim=config.hidden_dim,
                       output_dim=config.output_dim,
                       p=config.p,
                       device=DEVICE)

        model.train(train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_epochs=config.n_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    grad_clip=config.grad_clip,
                    wandb=wandb,
                    train_path=save_path,
                    t_sne=config.t_sne,
                    classifier=config.classifier,)
        
    else:
        #* clustering
        from clustering import tsne
        from TS2VEC import Encoder


        encoder = Encoder(input_dim=config.input_dim,
                               hidden_dim=config.hidden_dim,
                               output_dim=config.output_dim,
                               p=config.p).to(DEVICE)
        
        model = torch.optim.swa_utils.AveragedModel(encoder)

        # insert path to model_save
        t = args.model_path
        
        PATH = f"results/{config.dataset}/({t[1]}_{t[2]}_{t[3]})_({t[4]}_{t[5]}_{t[6]})"
        

        model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
        model.eval()

        
        # fig_train, fig_test = tsne(H=model, 
        #                     train_dataloader=train_dataloader, 
        #                     test_dataloader=test_dataloader, 
        #                     output_dim=args.output_dim[0], 
        #                     device=DEVICE,
        #                     save_path=PATH)





if any([len(i) > 1 for i in arguments.values() if type(i) == list]):
    print("initializing sweep")

    sweep_configuration = {
        "method": "grid",
        "metric": {"name": "tsloss/test_loss", "goal": "minimize"},
        "parameters": {
            i : {"values": j if type(j) == list else [j]}
            for (i,j) in arguments.items() 
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="BACHELOR_THESIS")

    wandb.agent(sweep_id, function=main)
else:
    # TODO: Arguments
    save = {}
    for (i,j) in arguments.items():
        if type(j) == list:
            save[i] = j[0]
        else:
            save[i] = j

    arguments = save
    print("initializing single value training")
    
    main(sweep=False)

