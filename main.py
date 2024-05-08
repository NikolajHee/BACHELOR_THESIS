"""
Main script of the Bachelor Thesis. 

The script can be executed in two ways:

1. As a sweep:
    - The script will execute the main function with all possible combinations of the arguments.
    - The sweep will be saved on wandb.

2. As a single value training:
    - The script will execute the main function with the given arguments.
"""



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
parser.add_argument('--dataset', default='PTB_XL') 
parser.add_argument('-id', '--input_dim', default=12, type=int)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-n', '--normalize', action='store_true')
parser.add_argument('--cda', action='store_true')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--t-sne', action='store_true')
parser.add_argument('--model_path', default=None, nargs='+')


# variables that can be sweeped
parser.add_argument('-c', '--classifier', default=None, choices=['logistic', 'svc', 'knn'], nargs='+')
parser.add_argument('-hd', '--hidden_dim', default=64, type=int, nargs='+')
parser.add_argument('-od', '--output_dim', default=320, type=int, nargs='+')
parser.add_argument('-bs', '--batch_size', default=8, type=int, nargs='+')
parser.add_argument('-ne', '--n_epochs', default=200, type=int, nargs='+')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, nargs='+')
parser.add_argument('-p', default=0.5, type=float, nargs='+')
parser.add_argument('-gc', '--grad_clip', default=None, type=float, nargs='+')
parser.add_argument('--N_train', default=None, type=int, nargs='+')
parser.add_argument('--N_test', default=None, type=int, nargs='+')
parser.add_argument('--alpha', default=0.5, type=float, nargs='+')

# manual features 
#   not implemented
parser.add_argument('--manual-features', action='store_true')


# get arguments from parser
args = parser.parse_args()
arguments = vars(args)


#* train function ; can be executed both by a sweep, or a by single values
def main(sweep=True):
    # initialize wandb (sweep or single value training)
    if sweep:
        print("initializing sweep")
        results_path = 'results/sweep'

        now = datetime.now()
        dt_string = now.strftime("(%d_%m_%Y)_(%H_%M_%S)")

        wandb.init(project="BACHELOR_THESIS",
                   name=dt_string)

    else:
        print("initializing single value training")
        results_path = 'results'

        now = datetime.now()
        dt_string = now.strftime("(%d_%m_%Y)_(%H_%M_%S)")

        wandb.init(project="BACHELOR_THESIS",
                   name=dt_string,
                   config=arguments)

    config = wandb.config

    test = vars(config)

    test2 = {i:j for (i,j) in test['_items'].items() if i!='_wandb'}

    #print(arguments)
    #print(test2)
    for (i,j) in test2.items():
        arguments[i] = j
    
    # remove lists
    for (i,j) in arguments.items():
        if type(j) == list:
            arguments[i] = j[0]

    #print(arguments)

    #raise NotImplementedError
    # save path
    save_path = os.path.join(results_path, arguments['dataset'], dt_string)

    # initialize time object
    time_object = TimeTaking(save_path=save_path)

    # create save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save parameters
    save_parameters(save_path, vars(args))

    # set random seed
    if arguments['seed'] is not None:
        random_seed(arguments['seed'])


    # load data
    if arguments['dataset'] == 'PTB_XL':
        from dataset import PTB_XL
        dataset = PTB_XL()
    else:
        from dataset import AEON_DATA
        # UCR and UEA datasets
        dataset = AEON_DATA(arguments['dataset'])

    print(f"using the data {arguments['dataset']}")

    # manual features
    #  not implemented
    # if config.manual_features:
    #     from manual_features import model
    #     return


    # create train/test-split
    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                     test_proportion=0.3,
                                                     train_size=arguments['N_train'],
                                                     test_size=arguments['N_test'],
                                                     return_stand=arguments['normalize'])
                                                    
    # Either train a model or load existing model
    if not arguments['model_path']:
        from TS2VEC import TS2VEC

        # initialize model
        model = TS2VEC(input_dim=arguments['input_dim'],
                       hidden_dim=arguments['hidden_dim'],
                       output_dim=arguments['output_dim'],
                       p=arguments['p'],
                       device=DEVICE)

        # start time object
        time_object.start('Model Training')

        # train the framework
        model.train(train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_epochs=arguments['n_epochs'],
                    batch_size=arguments['batch_size'],
                    learning_rate=arguments['learning_rate'],
                    grad_clip=arguments['grad_clip'],
                    alpha=arguments['alpha'],
                    wandb=wandb,
                    train_path=save_path,
                    t_sne=arguments['t_sne'],
                    classifier=arguments['classifier'],
                    )

        # end time object
        time_object.end('Model Training')
        time_object.save()

        if args.cda:
            from dataset import PTB_XL
            dataset = PTB_XL(multi_label=True)

            train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                     test_proportion=0.3,
                                                     train_size=arguments['N_train'],
                                                     test_size=arguments['N_test'],
                                                     return_stand=arguments['normalize'])

            from torch.utils.data import DataLoader

            train_dataloader = DataLoader(train_dataset, batch_size=arguments['batch_size'], shuffle=True, drop_last=True)
            test_dataloader = DataLoader(test_dataset, batch_size=arguments['batch_size'], shuffle=True, drop_last=True)

            from cda import cca
            cca(model=model.model,
                train_loader=train_dataloader,
                test_loader=test_dataloader,
                device=DEVICE,
                save_path=save_path)
        
    else:
        #* clustering
        from TS2VEC import Encoder


        encoder = Encoder(input_dim=arguments['input_dim'],
                               hidden_dim=arguments['hidden_dim'],
                               output_dim=arguments['output_dim'],
                               p=arguments['p']).to(DEVICE)
        
        model = torch.optim.swa_utils.AveragedModel(encoder)

        # insert path to model_save
        t = arguments['model_path']
        
        PATH = f"results/{arguments['dataset']}/({t[1]}_{t[2]}_{t[3]})_({t[4]}_{t[5]}_{t[6]})"
        

        model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
        model.eval()



# execute main function (sweep or single value training)
if any([len(i) > 1 for i in arguments.values() if type(i) == list]):
    print("initializing sweep")

    # # create sweep configuration (grid search)
    # sweep_configuration = {
    #     "method": "grid",
    #     "metric": {"name": "tsloss/hierarchical_contrastive_loss", "goal": "minimize"},
    #     "parameters": {
    #         i : {"values": j if type(j) == list else [j]}
    #         for (i,j) in arguments.items() 
    #     },
    # }

    sweep_configuration = {
        "method": "bayes",
        "name": "sweep_bayes_TS2VEC",
        "metric": {"name": "tsloss/hierarchical_contrastive_loss", "goal": "minimize"},
        "parameters": {
            i : {"min": j[0], "max": j[1]} for (i,j) in arguments.items() if (type(j) == list) and (len(j) == 2) and ((type(j[0]) is int) or (type(j[0]) is float))
        },
    }

    for (i,j) in arguments.items():
        if (type(j) == list) and (len(j) > 1) and ((type(j[0]) is not int) and (type(j[0]) is not float)):
            sweep_configuration['parameters'][i] = {'values': j}

    print(sweep_configuration)
    #print(sweep_configuration['parameters'])

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="BACHELOR_THESIS")

    wandb.agent(sweep_id, function=main)
else:
    # remove list from arguments (single value training)
    save = {}
    for (i,j) in arguments.items():
        if type(j) == list:
            save[i] = j[0]
        else:
            save[i] = j

    arguments = save
    print("initializing single value training")
    
    main(sweep=False)

