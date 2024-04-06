
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from code.scripts.utils import save_parameters
import wandb






np.random.seed(0)

parser = argparse.ArgumentParser(
                    prog='Main framework of Bachelor Thesis',
                    description='Representation learning and Machine Unlearning',
                    epilog='By Nikolaj Hertz s214644')

parser.add_argument('model') 
parser.add_argument('--dataset', default='PTB_XL') 
parser.add_argument('-sf', '--save_file', default='unnamed')
parser.add_argument('-c', '--classifier', default='logistic', choices=['logistic', 'svc'], nargs='+')
parser.add_argument('-od', '--output_dim', default=16, type=int, nargs='+')
parser.add_argument('-bs', '--batch_size', default=2, type=int, nargs='+')
parser.add_argument('-ne', '--n_epochs', default=4, type=int, nargs='+')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, nargs='+')
parser.add_argument('-p', default=0.5, type=float, nargs='+')
parser.add_argument('-id', '--input_dim', default=12, type=int)
parser.add_argument('-gc', '--grad_clip', default=0.01, type=float, nargs='+')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--N_train', default=10, type=int, nargs='+')
parser.add_argument('--N_test', default=10, type=int, nargs='+')


args = parser.parse_args()

arguments = vars(args)







def main():
    wandb.init(project="BACHELOR_THESIS")
        

    config = wandb.config
    # save_path = args.save_file
    # # add current date and time
    # now = datetime.now()

    # dt_string = now.strftime("%d_%m_%Y/%H_%M_%S")

    # save_path = os.path.join(save_path, dt_string)


    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # save_parameters(save_path, vars(args))





    if config.dataset == 'PTB_XL':
        from code.scripts.dataloader import PTB_XL
        dataset = PTB_XL()


    if config.model.lower() == 'ts2vec':
        from code.scripts.TS2VEC import train

        train_loss_save, test_loss_save, train_accuracy_save, test_accuracy_save, base = train(classifier=config.classifier,
                                                                                        dataset=dataset,
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
                                                                                        wandb=wandb)
        

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



if any([type(i) == list for i in arguments.values()]):

    sweep_configuration = {
        "method": "grid",
        "metric": {"name": "tsloss/test_loss", "goal": "minimize"},
        "parameters": {
            i : {"values": j if type(j) == list else [j]}
            for (i,j) in arguments.items() 
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="BACHELOR-THESIS")

    wandb.agent(sweep_id, function=main, count=10)

