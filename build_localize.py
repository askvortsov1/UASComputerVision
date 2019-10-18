import os
import sys
import time
import argparse 

import models 
import Datasets 
import utils.misc as misc
import utils.Meters as Meters 
import utils.nn_utils as nn_utils

import torch
import torch.optim as optim

model_options = sorted(name for name in models.__dict__
                       if not name.startswith("__")
                       and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()

""" ---- MODEL PARAMETERS ---- """
parser.add_argument('--model', default='yolov3', type=str, options=model_options)

""" ---- TRAINING PARAMETERS ---- """
parser.add_argument('--n_devices', default=4, type=int)
parser.add_argument('--use_cuda', default=1, type=int,
                    help='if cuda is unavailable but use_cuda is set to True, '
                         'the program will set use_cuda to False prior to training')
parser.add_argument('--n_threads', default=4, type=int)
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_test', default=0, type=int, help='if 0, will be set to value of batch_size')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--milestones', type=int, nargs='+', default=[60, 120, 160])
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--scheduler_type', default='multistep', type=str, choices=['multistep', 'cosine_annealing'])
parser.add_argument('--lr_min', default=0, type=float)
parser.add_argument('--epoch_reports', default=7, type=int, help='number of update displays per epoch')

""" ---- OPTIMIZER PARAMETERS ---- """
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--dampening', default=0, type=float)
parser.add_argument('--nesterov', default=1, type=int)

""" ---- DATA PARAMETERS ---- """
parser.add_argument('--dataset_loc', default='../data/generated_targers/simulated_fields', type=str)

args = parser.parse_args()

assert os.path.exists(args.dataset_loc)

""" DEFINE CUDA FUNCTIONALITY """
args.use_cuda = torch.cuda.is_available() & args.use_cuda
if args.use_cuda:
    cudnn.benchmark = True
    device = torch.device('cuda')
    args.n_devices = min(torch.cuda.device_count(), args.n_devices)

if not args.batch_size_test:
    args.batch_size_test = args.batch_size

""" CREATE MODEL NAME BASED ON MODEL ARCHITECTURE AND TRAINING PARAMS
A DIRECTORY OF NAME 'model_name' WILL BE CREATED AND TRAIN/TEST DATA WILL BE STORED HERE """
model_name = misc.name_model(args)

""" DEFINE CSV LOGGER AND ROWS TO WRITE DATA TO """
logger = misc.CSVLogger(filename='{name}/log.csv'.format(name=model_name), row=row, save=args.save)
if args.save:
    with open('{name}/parameters.txt'.format(name=model_name), 'w+') as f:
        f.write(str(args))

""" GET TRAINLOADER AND TESTLOADER FROM localize() """
trainloader, testloader = Datasets.localize_wrapper.localize(args)

""" BUILD MODEL AND CONVERT TO CUDA IF USING GPU & BUILD OPTIMIZER AND SCHEDULER """
model, optimizer, scheduler = nn_utils.build_neuralnet_components(args, n_batches=len(trainloader))

""" BUILD LOSS FUNCTION """
loss_fn = nn_utils.cost(args.use_cuda)


row = ['grid_size', 'loss', 'x', 'y', 'w', 'h', 'confidence', 'class', 'class_acc', \
    'recall50', 'recall75', 'precision', 'confidence_obj', 'confidence_noobj']


def train(epoch):
    model.train()
    batch = -1
    for batch_x, batch_y in enumerate(trainloader):
        batch += 1
        sz = batch_y.size(0)
        if args.use_cuda:
            batch_x, batch_y = batch_x.to(device, non_blocking=True, copy=False), \
                batch_y.to(device, non_blocking=True, copy=False)
            logits, loss = model(batch_x, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate():
    pass





