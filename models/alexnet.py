""" 
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
"""

import torch
import torch.nn as nn


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, high_res=1):
        super(AlexNet, self).__init__()
        k1 = 11 if high_res else 3
        s1 = 4 if high_res else 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=k1, stride=s1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(args):
    high_res = 0
    if args.dataset in ('imagenet', 'cub200', 'stanford_cars'):
        high_res = 1
    model = AlexNet(num_classes=args.n_classes, high_res=high_res)
    return model



