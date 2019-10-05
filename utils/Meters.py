import torch
import numpy as np
from time import time


class Timer():
    def __init__(self):
        self.o = time()

    def measure(self, p=1):
        x = (time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def accuracy(logits, labels, correct, total):
    _, pred = logits.max(1)
    total += labels.size(0)
    correct += pred.eq(labels).sum().item()
    return correct, total, 100.0 * correct/total


def accuracy_topk(logits, labels, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    
def calculate_time_remaining(t_init, batch, n_batches, epoch, n_epochs):
    epoch_time_diff = time() - t_init
    total = np.round((n_epochs * epoch_time_diff - ((epoch + ((batch + 1) / n_batches))
                                                    * epoch_time_diff)) / (epoch + ((batch + 1) / n_batches)))
    template = 'eta {}m {}s\n'.format(int(total // 60), int(total % 60))
    print(template)