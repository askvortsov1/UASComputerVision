import os
import torch
import pickle
import numpy as np 
from PIL import Image 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformations import FractionalMaxPool, AddBG



__all__ = ['shape', 'shape_color', 'letter', 'letter_color']


class FeatureDataset(Dataset):
    idxmap = {
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
        }}
    def __init__(self, root, feature, split='train', transform=None):
        super(Dataset, self).__init__()
        self.transform = transform 
        hashmap = self.idxmap[feature]
        with open(root, 'rb') as f:
            data = pickle.load(f)
        self.X = data['X']
        self.y = np.vectorize(lambda x: hashmap[x])(data[feature])
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img, label = self.X[idx], self.y[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(feature, split):
    # TODO CONFIGURE TRANSFORMATIONS SPECIFIC TO ATTRIBUTE BEING TRAINED FOR
    # if feature is not shape-color or letter-color, able to add in photometric transformations
    pass

    
def build_dataloaders(feature, batch_size, batch_size_test, n_threads): 
    # TODO CONFIGURE .PKL FILES WITH MAPPING FROM NUMPY IMAGE TO FEATURES
    return DataLoader(
        dataset=FeatureDataset(
            root='../../data/train_data.pkl', feature=feature, split='train', 
            transform=build_transforms(split='train')), batch_size=batch_size, 
            shuffle=True, num_workers=n_threads), \
                DataLoader(
                    dataset=FeatureDataset(
                        root='../../test_data.pkl', feature=feature, split='test', 
                        transform=build_transforms(split='test')), batch_size=batch_size_test,
                         shuffle=False, num_workers=n_threads)


def shape(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

def shape_color(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

def letter(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

def letter_color(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

        
        
    