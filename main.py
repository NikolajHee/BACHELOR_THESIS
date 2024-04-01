
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
                    prog='Main framework of Bachelor Thesis',
                    description='Representation learning and Machine Unlearning',
                    epilog='By Nikolaj Hertz s214644')

parser.add_argument('model') 
parser.add_argument('-sf', '--save_file', default='unnamed')
parser.add_argument('-v', '--verbose',
                    action='store_true')
parser.add_argument('-d', '--data', default='ptb_xl')



args = parser.parse_args()

save_path = args.save_file
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.model.lower() == 'ts2vec':
    from code.scripts.TS2VEC import train

    loss, class_loss = train(verbose=args.verbose, 
                 output_dim=256, 
                 batches=10, 
                 batch_size=100, 
                 n_epochs=20,
                 class_points=100)

    loss_mean = np.mean(loss, axis=1)

    plt.plot(loss_mean)
    plt.savefig(save_path + '/first_loss_function.png')
    plt.close()

    plt.plot(class_loss)
    plt.savefig(save_path + '/class_loss_function.png')
    plt.close()
    #train(args.save_file)

#print(args.filename, args.count, args.verbose)


# import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()

# print(args.accumulate(args.integers))
