import os

import torch
import torch.nn as nn
import torch.nn.functional as F 


__all__ = ['ssd']


class SSD(nn.Module):
    def __init__(self, args, phase):
        super(SSD, self).__init__()
        self.phase = phase
        self.n_classes = args.n_classes
        


def ssd(args, phase):
    return SSD
