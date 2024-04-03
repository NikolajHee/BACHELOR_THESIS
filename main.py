
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from code.scripts.utils import save_parameters


parser = argparse.ArgumentParser(
                    prog='Main framework of Bachelor Thesis',
                    description='Representation learning and Machine Unlearning',
                    epilog='By Nikolaj Hertz s214644')

parser.add_argument('model') 
parser.add_argument('--dataset', default='PTB_XL') 
parser.add_argument('-sf', '--save_file', default='unnamed')
parser.add_argument('-c', '--classifier', default='logistic', choices=['logistic', 'svc'])
parser.add_argument('-od', '--output_dim', default=256, type=int)
parser.add_argument('-bs', '--batch_size', default=8, type=int)
parser.add_argument('-ne', '--n_epochs', default=4, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('-p', default=0.5, type=float)
parser.add_argument('-id', '--input_dim', default=12, type=int)
parser.add_argument('-gc', '--grad_clip', default=0.01, type=float)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--ts_num_batches', default=[2,2], type=int, nargs='+')
parser.add_argument('--class_num_batches', default=[2,2], type=int, nargs='+')


args = parser.parse_args()





save_path = args.save_file
# add current date and time
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y/%H_%M_%S")

save_path = os.path.join(save_path, dt_string)


if not os.path.exists(save_path):
    os.makedirs(save_path)

save_parameters(save_path, vars(args))

if args.dataset == 'PTB_XL':
    from code.scripts.dataloader import PTB_XL
    dataset = PTB_XL()


if args.model.lower() == 'ts2vec':
    from code.scripts.TS2VEC import train

    train_loss_save, test_loss_save, train_accuracy_save, test_accuracy_save = train(classifier=args.classifier,
                                                                                    dataset=dataset,
                                                                                    output_dim=args.output_dim,
                                                                                    n_epochs=args.n_epochs,
                                                                                    batch_size=args.batch_size,
                                                                                    learning_rate=args.learning_rate,
                                                                                    p=args.p,
                                                                                    input_dim=args.input_dim,
                                                                                    grad_clip=args.grad_clip,
                                                                                    verbose=args.verbose,
                                                                                    ts_num_batches=tuple(args.ts_num_batches),
                                                                                    class_num_batches=tuple(args.class_num_batches))
    

    train_loss_mean = np.mean(train_loss_save, axis=1)
    test_loss_mean = np.mean(test_loss_save, axis=1)

    np.save(os.path.join(save_path, 'train_ts2vec_loss'), train_loss_mean)
    np.save(os.path.join(save_path, 'test_ts2vec_loss'), test_loss_mean)
    np.save(os.path.join(save_path, 'train_accuracy_save'), train_accuracy_save)
    np.save(os.path.join(save_path, 'test_accuracy_save'), test_accuracy_save)


    # plt.plot(loss_mean)
    # plt.savefig(save_path + '/first_loss_function.png')
    # plt.close()

    # plt.plot(class_loss)
    # plt.savefig(save_path + '/class_loss_function.png')
    # plt.close()


