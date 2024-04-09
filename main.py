
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import save_parameters
import wandb
import torch






np.random.seed(0)

parser = argparse.ArgumentParser(
                    prog='Main framework of Bachelor Thesis',
                    description='Representation learning and Machine Unlearning',
                    epilog='By Nikolaj Hertz s214644')

parser.add_argument('model') 
parser.add_argument('--dataset', default='PTB_XL') 
parser.add_argument('-sf', '--save_file', default='unnamed')
parser.add_argument('-c', '--classifier', default='logistic', choices=['logistic', 'svc'], nargs='+')
parser.add_argument('-hd', '--hidden_dim', default=64, type=int, nargs='+')
parser.add_argument('-od', '--output_dim', default=320, type=int, nargs='+')
parser.add_argument('-bs', '--batch_size', default=8, type=int, nargs='+')
parser.add_argument('-ne', '--n_epochs', default=200, type=int, nargs='+')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, nargs='+')
parser.add_argument('-p', default=0.5, type=float, nargs='+')
parser.add_argument('-id', '--input_dim', default=12, type=int)
parser.add_argument('-gc', '--grad_clip', default=None, type=float, nargs='+')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--N_train', default=None, type=int, nargs='+')
parser.add_argument('--N_test', default=None, type=int, nargs='+')


debug_mode = False

args = parser.parse_args()

arguments = vars(args)







def main():
    config = wandb.config
    save_path = config.save_file
    # # add current date and time


    save_path = os.path.join(save_path, dt_string)


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save_parameters(save_path, vars(args))





    if config.dataset == 'PTB_XL':
        from dataloader import PTB_XL
        dataset = PTB_XL()
    elif config.dataset == 'ElectricDevices':
        from dataloader import UEA
        dataset = UEA(config.dataset)
        print(dataset)


    if config.model.lower() == 'ts2vec':
        from TS2VEC import train

        train_loss_save, test_loss_save, train_accuracy_save, test_accuracy_save, base = train(classifier=config.classifier,
                                                                                                dataset=dataset,
                                                                                                hidden_dim=config.hidden_dim,
                                                                                                output_dim=config.output_dim,
                                                                                                n_epochs=config.n_epochs,
                                                                                                batch_size=config.batch_size,
                                                                                                learning_rate=config.learning_rate,
                                                                                                p=config.p,
                                                                                                input_dim=config.input_dim,
                                                                                                grad_clip=config.grad_clip,
                                                                                                verbose=config.verbose,
                                                                                                N_train=config.N_train,
                                                                                                N_test=config.N_test,
                                                                                                wandb=wandb,
                                                                                                train_path=save_path)
        

        # np.save(os.path.join(save_path, 'train_ts2vec_loss'), train_loss_save)
        # np.save(os.path.join(save_path, 'test_ts2vec_loss'), test_loss_save)
        # np.save(os.path.join(save_path, 'train_accuracy_save'), train_accuracy_save)
        # np.save(os.path.join(save_path, 'test_accuracy_save'), test_accuracy_save)
        # np.save(os.path.join(save_path, 'baseline'), np.ones(len(train_accuracy_save)) * base)
        

        # plt.plot(loss_mean)
        # plt.savefig(save_path + '/first_loss_function.png')
        # plt.close()

        # plt.plot(class_loss)
        # plt.savefig(save_path + '/class_loss_function.png')
        # plt.close()


if any([len(i) > 1 for i in arguments.values() if type(i) == list]):



    sweep_configuration = {
        "method": "grid",
        "metric": {"name": "tsloss/test_loss", "goal": "minimize"},
        "parameters": {
            i : {"values": j if type(j) == list else [j]}
            for (i,j) in arguments.items() 
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="BACHELOR_THESIS")

    wandb.agent(sweep_id, function=main, count=10)
else:
    save = {}
    for (i,j) in arguments.items():
        if type(j) == list:
            save[i] = j[0]
        else:
            save[i] = j
    now = datetime.now()

    dt_string = now.strftime("(%d_%m_%Y)_(%H_%M_%S)")

    wandb.init(project="BACHELOR_THESIS",
               name=dt_string,
               config=save)
    
    main()

