import os
import argparse
import numpy as np 
from time import time 
from tqdm import tqdm 

import NN as models
import OPTIM as optimizers
import Datasets as datasets

import utils.misc as misc
import utils.Meters as Meters

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR

dataset_options = sorted(name for name in datasets.__dict__ 
                        if not name.startswith("__") 
                        and callable(datasets.__dict__[name]))

model_options = sorted(name for name in models.__dict__
                        if not name.startswith("__")
                        and callable(models.__dict__[name]))

optimizer_options = sorted(name for name in optimizers.__dict__
                        if not name.startswith("__")
                        and callable(optimizers.__dict__[name]))


parser = argparse.ArgumentParser()
parser.add_argument('--save', default=0, type=int)
parser.add_argument('--use_cuda', default=1, type=int)
parser.add_argument('--n_devices', default=4, type=int)
parser.add_argument('--subdir', default='tests', type=str)
parser.add_argument('--reload', default=1, type=int,
                    help='set to 1 to retransform training instances over every epoch')

parser.add_argument('--dataset', default='localize', choices=dataset_options, type=str)
parser.add_argument('--n_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_test', default=0, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 225])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--scheduler_type', default='multistep', type=str, choices=['multistep', 'cosine_annealing'])
parser.add_argument('--lr_min', default=0, type=float)
parser.add_argument('--epoch_reports', default=15, type=int, help='number of update displays per epoch')

parser.add_argument('--model', default='', choices=model_options, type=str)
parser.add_argument('--optimizer', default='sgd', choices=optimizer_options, type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--dampening', default=0, type=float)
parser.add_argument('--nesterov', default=1, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--amsgrad', default=0, type=int)
parser.add_argument('--rho', default=0.9, type=float, help='only considered when using Adadelta')
parser.add_argument('--lr_decay', default=0.0, type=float, help='only considered when using Adagrad')
parser.add_argument('--centered', default=0, type=int, help='only considered when using RMSprop')

parser.add_argument('--resolution', type=int, default=2048,
                    help='set if trying to fit to required net dimensions or if reducing dimensionality')


class train_eval:
    def __init__(self):
        if args.save:
            self.model_name = misc.name_model(args)
            with open('{name}/parameters.txt'.format(name=self.model_name), 'w+') as f:
                f.write(str(args))
            self.log_path = '{name}/log.txt'.format(name=self.model_name)
        else:
            self.log_path = './log.txt'
        misc.log(self.log_path, str(vars(args)))
        trainloader, self.testloader = datasets.__dict__[args.dataset](args, train=True, test=True)
        if not args.reload:
            self.trainloader = trainloader
        self.n_batches = len(trainloader)
        self.n_batches_eval = len(self.testloader)


def main():
    prog = train_eval()
    trainloader, testloader = datasets.__dict__[args.dataset](args, train=True, test=True)
    for x, y in trainloader:
        print(x, y)
        import matplotlib.pyplot as plt 
        plt.imshow(np.transpose(x[0].numpy(), (1, 2, 0)))
        plt.show()
        quit()


if __name__ == "__main__":
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available() & args.use_cuda
    args.n_devices = min(torch.cuda.device_count(), args.n_devices)
    if not args.batch_size_test:
        args.batch_size_test = args.batch_size
    main()