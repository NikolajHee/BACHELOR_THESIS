
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
parser.add_argument('-sf', '--save_file', default='unnamed')
parser.add_argument('-c', '--classifier', default='logistic', choices=['logistic', 'svc'])
parser.add_argument('-od', '--output_dim', default=256, type=int)
parser.add_argument('-b', '--batches', default=2, type=int)
parser.add_argument('-bs', '--batch_size', default=8, type=int)
parser.add_argument('-ne', '--n_epochs', default=4, type=int)
parser.add_argument('--class_points', default=64, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('-p', default=0.5, type=float)
parser.add_argument('-id', '--input_dim', default=12, type=int)
parser.add_argument('-gc', '--grad_clip', default=0.01, type=float)
parser.add_argument('-v', '--verbose', action='store_true')



args = parser.parse_args()





save_path = args.save_file
# add current date and time
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y/%H_%M_%S")

save_path = os.path.join(save_path, dt_string)


if not os.path.exists(save_path):
    os.makedirs(save_path)

save_parameters(save_path, vars(args))

if args.model.lower() == 'ts2vec':
    from code.scripts.TS2VEC import train

    loss, class_loss = train(classifier=args.classifier,
                             output_dim=args.output_dim,
                             batches=args.batches,
                             n_epochs=args.n_epochs,
                             batch_size=args.batch_size,
                             class_points=args.class_points,
                             learning_rate=args.learning_rate,
                             p=args.p,
                             input_dim=args.input_dim,
                             grad_clip=args.grad_clip,
                             verbose=args.verbose)
    

    loss_mean = np.mean(loss, axis=1)

    np.save(os.path.join(save_path, 'ts2vec_loss'), loss_mean)
    np.save(os.path.join(save_path, 'class_loss'), class_loss)

    # plt.plot(loss_mean)
    # plt.savefig(save_path + '/first_loss_function.png')
    # plt.close()

    # plt.plot(class_loss)
    # plt.savefig(save_path + '/class_loss_function.png')
    # plt.close()


