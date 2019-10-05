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


parser = argparse.ArgumentParser(description='Classification Training')
parser.add_argument('--save', default=0, type=int)
parser.add_argument('--use_cuda', default=1, type=int)
parser.add_argument('--n_devices', default=4, type=int)
parser.add_argument('--subdir', default='tests', type=str)
parser.add_argument('--message', default='', type=str)
parser.add_argument('--datadir', default='./data/generated_targets/60000/', type=str)
parser.add_argument('--csv_train', default='./data/CSV/train50000_csv.csv')
parser.add_argument('--csv_test', default='./data/CSV/test10000_csv.csv')

# parser.add_argument('--dataset', required=True, choices=dataset_options, type=str)
parser.add_argument('--dataset', default='shape', choices=dataset_options, type=str)
parser.add_argument('--n_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_test', default=0, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 225])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--scheduler_type', default='cosine_annealing', type=str, choices=['multistep', 'cosine_annealing'])
parser.add_argument('--lr_min', default=0, type=float)
parser.add_argument('--epoch_reports', default=15, type=int, help='number of update displays per epoch')

parser.add_argument('--model', default='resnetx', choices=model_options, type=str)
parser.add_argument('--pretrained', default=0, type=int)
parser.add_argument('--depth', default=20, type=int)
parser.add_argument('--width', default=10, type=int, help='only considered if using WideResNet')
parser.add_argument('--drop_prob', default=0.0, type=float)
# parser.add_argument('--n_classes', required=True, type=int)
parser.add_argument('--n_classes', default=13, type=int)

parser.add_argument('--base_channels', default=16, type=int)
parser.add_argument('--shake_forward', default=1, type=int)
parser.add_argument('--shake_backward', default=1, type=int)
parser.add_argument('--shake_image', default=1, type=int)

parser.add_argument('--growth_rate', default=40, type=int, help='only considered if using densenetBC')
parser.add_argument('--reduce', default=0.5, type=float, help='only considered if using densenetBC')
parser.add_argument('--bottleneck', type=int, default=1, help='only considered if using densenetBC')
parser.add_argument('--alpha', default=270, type=int, help='only considered if using pyramidnet')
parser.add_argument('--x', default=1, type=int, help='only considered if using resnetx')

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

parser.add_argument('--loss_fn', default='nlogsoftmax', choices=['cross_entropy', 'nlogsoftmax'])

parser.add_argument('--resolution', type=int, default=8,
                    help='set if trying to fit to required net dimensions or if reducing dimensionality')
parser.add_argument('--mpp', type=float, default=0.0, help='fractional maxpool probability')
parser.add_argument('--reload', default=1, type=int,
                    help='set to 1 to retransform training instances over every epoch')
parser.add_argument('--bg_prob', default=1.0, type=float, help='probabiltiy of adding a background to the image')
parser.add_argument('--fmp_prob', default=0.8, type=float,
                    help='probability of augmenting the training image by fractional max pooling')
parser.add_argument('--min_resize_ratio', default=0.5, type=float)
parser.add_argument('--kernel_sizes', default=[2, 0.4, 3, 0.55, 4, 0.05], nargs='+', type=float,
                    help='kernel size probability distribution for fractional max pooling')


class Util:
    def __init__(self):
        self.stats = argparse.Namespace(**{'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []})

    @staticmethod
    def get_meters(n_batches, epoch=None):
        losses = Meters.AverageMeter('Loss', ':.2e')
        acc = Meters.AverageMeter('Acc', ':6.2f')
        meters = [acc, losses]
        return meters
    
    @staticmethod
    def build_neuralnet():
        net = models.__dict__[args.model](args)
        if args.use_cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(args.n_devices))
        return net

    @staticmethod
    def cost():
        if args.loss_fn == 'cross_entropy':
            return nn.CrossEntropyLoss().cuda() if args.use_cuda else nn.CrossEntropyLoss()
        elif args.loss_fn == 'nlogsoftmax':
            if args.use_cuda:
                return lambda logits, targets: (-F.log_softmax(logits, dim=1) * 
                                                torch.zeros(logits.size()).cuda().scatter_(1, targets.data.view(-1, 1),
                                                torch.ones(size=(logits.size(0), )).view(-1, 1))).sum(dim=1).mean()
            else:
                return lambda logits, targets: (-F.log_softmax(logits, dim=1) * 
                                                torch.zeros(logits.size()).scatter_(1, targets.data.view(-1, 1),
                                                torch.ones(size=(logits.size(0), )).view(-1, 1))).sum(dim=1).mean()
    
    @staticmethod
    def build_optimizer(net, n_batches):
        scheduler = 0
        def get_cosine_annealing_scheduler(lr_init, lr_min, n_epochs):
            def _cosine_annealing(step, total_steps, lr_max, lr_min):
                return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
            total_steps = n_epochs * n_batches
            scheduler = LambdaLR(
                optimizer, lr_lambda=lambda step: _cosine_annealing(step, total_steps, 1, lr_min / lr_init))
            return scheduler
        optimizer = optimizers.__dict__[args.optimizer](net.parameters(), args)
        if args.scheduler_type == 'multistep':
            assert sum(args.milestones) != 0
            scheduler = MultiStepLR(optimizer, milestones=np.array(args.milestones) * n_batches, gamma=args.gamma)
        elif args.scheduler_type == 'cosine_annealing':
            scheduler = get_cosine_annealing_scheduler(lr_init=args.lr, lr_min=args.lr_min, n_epochs=args.n_epochs)
        return optimizer, scheduler
    
    @staticmethod
    def network_summary(net, input_shape, batch_size, device):
        summary(net, input_shape, batch_size, device)


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
        self.net = util.build_neuralnet()
        Util.network_summary(net=self.net, input_shape=(3, args.resolution, args.resolution), batch_size=args.batch_size, device='cuda' if args.use_cuda else 'cpu')
        self.loss_fn = util.cost()
        self.optimizer, self.scheduler = util.build_optimizer(net=self.net, n_batches=self.n_batches)
        self.best_acc = -np.inf
        self.timer = Meters.Timer()

    def run_train(self):
        final_epoch = False
        for epoch in range(1, args.n_epochs + 1):
            misc.log(self.log_path, 'Elapsed Time: {}/{}\n'.format(
                self.timer.measure(), self.timer.measure(epoch / float(args.n_epochs))))
            if self.scheduler:
                lr = self.scheduler.get_lr()[0]
            else:
                lr = args.lr
            self.train(epoch)
            acc = self.evaluate()
            improvement = acc > self.best_acc
            self.best_acc = max(acc, self.best_acc)
            misc.log(self.log_path, 'Best Accuracy: {} | Current Learning Rate: {}'.format(np.round(self.best_acc, 5), np.round(lr, 5)))
            if epoch == args.n_epochs:
                final_epoch = True
            if args.save:
                misc.save_model(args=args, model_name=self.model_name, best_acc=self.best_acc, stats=util.stats,
                                state={'epoch': epoch, 'state_dict': self.net.state_dict(),
                                       'best_acc': self.best_acc, 'optimizer': self.optimizer.state_dict()},
                                improvement=improvement, epoch=epoch, final_epoch=final_epoch)

    def train(self, epoch):
        if args.reload:
            trainloader = datasets.__dict__[args.dataset](args, train=True, test=False)
        else:
            trainloader = self.trainloader
        acc, losses = util.get_meters(self.n_batches, epoch=epoch)
        print_mod = int(self.n_batches / args.epoch_reports)
        self.net.train()
        batch = -1
        for inputs, labels in tqdm(trainloader, total=self.n_batches):
            batch += 1
            """
            from imageio import imwrite
            import torchvision
            imwrite(uri='./data/train_instance8x8.png', im=np.transpose(torchvision.utils.make_grid(inputs, nrow=16, padding=0).numpy(), (1, 2, 0)))
            quit()
            """
            if args.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            logits = self.net(inputs)
            loss = self.loss_fn(logits, labels)
            _, pred = logits.max(1)
            correct = pred.eq(labels).sum().item()
            batch_acc = 100. * correct / labels.size(0)
            acc.update(batch_acc, 1)
            losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            util.stats.train_acc.append(batch_acc)
            util.stats.train_loss.append(loss.item())
            if batch % print_mod == print_mod - 1 or batch == self.n_batches - 1:
                log_str = '\nEpoch: [{}/{}]\tBatch: [{}/{}]\tLoss: {:.4f}\tAcc: {:.2f} % ({:.2f} %)'.format(
                    epoch, args.n_epochs, batch, self.n_batches, loss.item(), batch_acc, acc.avg)
                misc.log(self.log_path, log_str)
        
    def evaluate(self):
        acc, losses = util.get_meters(self.n_batches_eval, epoch=None)
        print_mod = int(self.n_batches_eval / args.epoch_reports)
        count = total = 0
        self.net.eval()
        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(tqdm(self.testloader), 1):
                if args.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                logits = self.net(inputs)
                loss = self.loss_fn(logits, labels)
                losses.update(loss.item(), inputs.size(0))
                _, pred = logits.max(1)
                correct = pred.eq(labels).sum().item()
                batch_acc = 100. * correct / labels.size(0)
                count += correct
                total += labels.size(0)
                acc.update(batch_acc, labels.size(0))
                util.stats.test_loss.append(loss.item())
                util.stats.test_acc.append(batch_acc)
                if batch % print_mod == print_mod - 1:
                    log_str = '\nBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                        batch, self.n_batches_eval, loss.item(), batch_acc, acc.avg)
                    misc.log(self.log_path, log_str)
            print('Acc {acc.avg:.3f} | Loss {losses.avg:.2e}'.format(acc=acc, losses=losses))
        return 100. * count / total


def main():
    prog = train_eval()
    prog.run_train()

    
if __name__ == '__main__':
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available() & args.use_cuda
    args.n_devices = min(torch.cuda.device_count(), args.n_devices)
    if not args.batch_size_test:
        args.batch_size_test = args.batch_size
    util = Util()
    main()
