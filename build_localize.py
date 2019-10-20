import os
import sys
import time
import argparse 

import models 
import utils.misc as misc
import utils.Meters as Meters 
from Datasets.localize_wrapper import localize_dataset
# from SingleShotDetection import SSD300, MultiBoxLoss

import torch
import torch.optim as optim
import torchvision.transforms as transforms

dataset_options = [dataset for dataset in os.listdir('../data/Aerial/datasets') if dataset != '.DS_Store']

model_options = sorted(name for name in models.__dict__
                       if not name.startswith("__")
                       and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()

""" ---- SAVING CONFIG ---- """
parser.add_argument('--no-save', action='store_false', dest='save')
parser.add_argument('--message', default='', type=str, help='add optional message to directory name')

""" ---- MODEL PARAMETERS ---- """
parser.add_argument('--model', default='yolov3', type=str, choices=model_options)
parser.add_argument('--n_classes', default=13, type=int)
parser.add_argument('--resolution', default=256, type=int)

""" ---- TRAINING PARAMETERS ---- """
parser.add_argument('--n_devices', default=4, type=int)
parser.add_argument('--no-use_cuda', action='store_false', dest='use_cuda')
parser.add_argument('--n_processes', default=4, type=int)
parser.add_argument('--dataset', default=dataset_options[0], type=str)
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--batch_size_test', default=8, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--milestones', type=int, nargs='+', default=[60, 120, 160])
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--scheduler_type', default='multistep', type=str, choices=['multistep', 'cosine_annealing'])
parser.add_argument('--lr_min', default=0, type=float)
parser.add_argument('--epoch_reports', default=7, type=int, help='number of update displays per epoch')

""" ---- OPTIMIZER PARAMETERS ---- """
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--dampening', default=0, type=float)
parser.add_argument('--no-nesterov', action='store_false', dest='nesterov')

args = parser.parse_args()
print(args.dataset)

""" DEFINE CUDA FUNCTIONALITY """
args.use_cuda = torch.cuda.is_available() & args.use_cuda
if args.use_cuda:
    cudnn.benchmark = True
    device = torch.device('cuda')
    args.n_devices = min(torch.cuda.device_count(), args.n_devices)

""" CREATE MODEL NAME BASED ON MODEL ARCHITECTURE AND TRAINING PARAMS
A DIRECTORY OF NAME 'model_name' WILL BE CREATED AND TRAIN/TEST DATA WILL BE STORED HERE """
model_name = misc.name_model(args)

""" DEFINE CSV LOGGER AND ROWS TO WRITE DATA TO 
logger = misc.CSVLogger(filename='{name}/log.csv'.format(name=model_name), row=row, save=args.save)
if args.save:
    with open('{name}/parameters.txt'.format(name=model_name), 'w+') as f:
        f.write(str(args))"""

""" GET TRAINLOADER AND TESTLOADER FROM localize() """

trainloader, testloader = localize_dataset(root=args.dataset, batch_size=args.batch_size, \
    batch_size_test=args.batch_size_test, n_processes=args.n_processes, resolution=args.resolution, \
        transform=transforms.Compose([transforms.ToTensor()]))

""" BUILD MODEL AND CONVERT TO CUDA IF USING GPU & BUILD OPTIMIZER AND SCHEDULER """
model = SSD300(n_classes=args.n_classes)
biases = list()
not_biases = list()
for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)

optimizer = optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr}, {'params': not_biases}], lr=args.lr, \
    momentum=args.momentum, weight_decay=args.weight_decay)
model = model.to(device) 
loss_fn = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

epochs_since_imp = 0

def train(epoch):
    model.train()
    n_batches = len(trainloader)
    for i, (images, boxes, labels) in enumerate(trainloader):
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = loss_fn(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()


        # Print status
        if i % 10 == 0:
            print('batch: [{}/{}]'.format(i, n_batches))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(trainloader),
                                                                  loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels


def validate():
    model.eval()  # eval mode disables dropout
    n_batches = len(testloader)
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(testloader):

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = loss_fn(predicted_locs, predicted_scores, boxes, labels)


            # Print status
            if i % 10 == 0:
                print('batch: [{}/{}]'.format(i, n_batches))
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


for epoch in range(1, args.n_epochs + 1):

        # One epoch's training
        train(epoch)

        # One epoch's validation
        val_loss = validate()

        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_imp))

        else:
            epochs_since_improvement = 0




"""
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
"""




