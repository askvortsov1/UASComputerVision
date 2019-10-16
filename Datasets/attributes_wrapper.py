import os
import h5py
import torch
import numpy as np 
from PIL import Image 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformations import FractionalMaxPool, AddBG


__all__ = ['shape', 'shape_color', 'letter', 'letter_color']


class FeatureDataset(Dataset):
    
    def __init__(self, root, feature, split='train', transform=None):
        super(Dataset, self).__init__()
        self.transform = transform 
        file = h5py.File(root, "r+")
        self.X = np.array(file["/X"]) / 255
        self.y = np.array(file["/{feature}".format(feature=feature)]).astype('uint8')[:, 0]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img, label = self.X[idx], self.y[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(feature, split, transform_params):
    # TODO CONFIGURE TRANSFORMATIONS SPECIFIC TO ATTRIBUTE BEING TRAINED FOR
    # if feature is not shape-color or letter-color, able to add in photometric transformations
    addbg = AddBG(root=transform_params['background_root'], bg_prob=transform_params['bg_prob'], \
        min_resize_ratio=transform_params['min_resize_ratio'], res=transform_params['resolution'])
    fmp = FractionalMaxPool()
    transform = transforms.Compose([
        addbg, fmp, transforms.ToTensor()
    ])
    return transform

    
def build_dataloaders(feature, batch_size, batch_size_test, n_threads): 
    transform_params={'background_root': '../../data/Aerial/100x100masati-v1.h5', 
                      'bg_prob': 1.0, 'min_resize_ratio': 0.5, 'resolution': (32, 32)}
    transform = build_transforms('_', '_', transform_params=transform_params)
    return DataLoader(
        dataset=FeatureDataset(
            root='../../data/generated_targets/train_data32x32.h5', feature=feature, split='train', 
            transform=transform), batch_size=batch_size, 
            shuffle=True, num_workers=n_threads), \
                DataLoader(
                    dataset=FeatureDataset(
                        root='../../data/generated_targets/test_data32x32.h5', feature=feature, split='test', 
                        transform=transform), batch_size=batch_size_test,
                         shuffle=False, num_workers=n_threads)


def shape(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

def shape_color(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

def letter(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

def letter_color(feature, batch_size, batch_size_test, n_threads):
    return build_dataloaders(feature, batch_size, batch_size_test, n_threads)

        
show_batch = False
if show_batch: 
    train, test = shape('shape', 64, 16, 1)
    import matplotlib.pyplot as plt 
    import torchvision
    d_iter = iter(train)
    for x, y in d_iter:

        plt.imshow(np.transpose(torchvision.utils.make_grid(x, nrow=8, padding=0).numpy(), (1, 2, 0)))
        plt.show()
        print(y)
        break

    