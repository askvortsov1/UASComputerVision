import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.transformations import FractionalMaxPool, AddBackground

__all__ = ['shape', 'shape_color', 'letter', 'letter_color']


class IMAGEDATASET(data.Dataset):
    idx_map = {
        'shape': {
            'circle': 0, 'semicircle': 1, 'quarter_circle': 2, 'triangle': 3, 'square': 4, 'rectangle': 5,
            'trapezoid': 6, 'pentagon': 7, 'hexagon': 8, 'heptagon': 9, 'octagon': 10, 'star': 11, 'cross': 12
        },
        'color': {'red': 0, 'orange': 1, 'yellow': 2, 'green': 3, 'blue': 4,
                  'purple': 5, 'brown': 6, 'gray': 7, 'white': 8},
        'letter': {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
            'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
            'X': 22, 'Y': 23, 'Z': 24, '1': 25, '2': 26, '3': 27, '4': 28, '5': 29, '6': 30, '7': 31, '8': 32, '0': 33
        }
    }

    def __init__(self, args, func_name, transforms=None, train=True):

        assert os.path.exists(args.datadir + 'train') and \
               os.path.exists(args.datadir + 'test') and \
               os.path.exists(args.csv_train) and os.path.exists(args.csv_test)
        if func_name in ('shape_color', 'letter_color'):
            func_name = 'color'
        mapping = self.idx_map[func_name]
        self.transforms = transforms
        if train:
            dataframe = pd.read_csv(args.csv_train)
        else:
            dataframe = pd.read_csv(args.csv_test)
        self.img_paths = dataframe['path'].values
        self.targets = np.vectorize(lambda x: mapping[x])(dataframe[func_name].values)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        target = self.targets[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


def shape(args, train, test):
    return_arr = []
    if train:
        transform_train = transforms.Compose([
            lambda x: np.array(x) / 255,
            AddBackground(args),
            FractionalMaxPool(args),
            transforms.Resize(size=args.resolution),
            transforms.ToTensor()])
        trainset = IMAGEDATASET(args=args, func_name=shape.__name__,
                                transforms=transform_train, train=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)

        return_arr.append(trainloader)

    if test:
        transform_test = transforms.Compose([transforms.Resize(size=args.resolution),
                                             transforms.ToTensor()])
        testset = IMAGEDATASET(args=args, func_name=shape.__name__,
                                transforms=transform_test, train=False)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test,
                                                shuffle=False, num_workers=2)
        return_arr.append(testloader)
    if len(return_arr) == 1:
        return return_arr[0]
    return return_arr


def shape_color(args, train, test):
    return_arr = []
    if train:
        transform_train = transforms.Compose([
            lambda x: np.array(x) / 255,
            AddBackground(args),
            FractionalMaxPool(args),
            transforms.Resize(size=args.resolution),
            transforms.ToTensor()])
        trainset = IMAGEDATASET(args=args, func_name=shape_color.__name__,
                                transforms=transform_train, train=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)
        return_arr.append(trainloader)

    if test:
        transform_test = transforms.Compose([
            lambda x: np.array(x) / 255,
            AddBackground(args),
            transforms.Resize(size=args.resolution),
            transforms.ToTensor()])
        testset = IMAGEDATASET(args=args, func_name=shape_color.__name__,
                               transforms=transform_test, train=False)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test,
                                                shuffle=False, num_workers=2)
        return_arr.append(testloader)
    if len(return_arr) == 1:
        return return_arr[0]
    return return_arr


def letter(args, train, test):
    return_arr = []
    if train:
        transform_train = transforms.Compose([
            lambda x: np.array(x) / 255,
            AddBackground(args),
            FractionalMaxPool(args),
            transforms.Resize(size=args.resolution),
            transforms.ToTensor()])
        trainset = IMAGEDATASET(args=args, func_name=letter.__name__,
                                transforms=transform_train, train=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)
        return_arr.append(trainloader)

    if test:
        transform_test = transforms.Compose([
            transforms.Resize(size=args.resolution),
            transforms.ToTensor()])
        testset = IMAGEDATASET(args=args, func_name=letter.__name__,
                               transforms=transform_test, train=False)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test,
                                                shuffle=False, num_workers=2)
        return_arr.append(testloader)
    if len(return_arr) == 1:
        return return_arr[0]
    return return_arr


def letter_color(args, train, test):
    return_arr = []
    if train:
        transform_train = transforms.Compose([
            lambda x: np.array(x) / 255,
            AddBackground(args),
            FractionalMaxPool(args),
            transforms.Resize(size=args.resolution),
            transforms.ToTensor()])
        trainset = IMAGEDATASET(args=args, func_name=letter_color.__name__,
                                transforms=transform_train, train=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)
        return_arr.append(trainloader)

    if test:
        transform_test = transforms.Compose([
            transforms.Resize(size=args.resolution),
            transforms.ToTensor()])
        testset = IMAGEDATASET(args=args, func_name=letter_color.__name__,
                               transforms=transform_test, train=False)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test,
                                                shuffle=False, num_workers=2)
        return_arr.append(testloader)
    if len(return_arr) == 1:
        return return_arr[0]
    return return_arr
