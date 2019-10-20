import os
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from Datasets.transformations import FractionalMaxPool
from torch.utils.data import Dataset, DataLoader


def show_bounding_boxes(img, boxes):
    import matplotlib.pyplot as plt 
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(20, 20)
    ax[0].imshow(img)
    ax[1].imshow(img)
    for box in boxes:
        box = (box * img.size[0]).numpy().astype('uint32')
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=0.75, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
    plt.show()


def show_sample_batch():
    import matplotlib.pyplot as plt 
    import torchvision
    root = 'train-10000_test-2000_res-256_lower-10_upper-24'
    batch_size = 16
    batch_size_test = 4
    n_threads = 1
    transform = transforms.Compose([transforms.ToTensor()])
    train, test = localize_dataset(root, batch_size, batch_size_test, transform, n_threads)
    X, y = iter(train).__next__()
    plt.imshow(np.transpose(torchvision.utils.make_grid(X, nrow=4, padding=0).numpy(), (1, 2, 0)))
    plt.show()
    

class ObjectDetection(Dataset):
    dataset_folder = '../data/Aerial/datasets'
    def __init__(self, root, split='train', transform=None, resolution=256):
        """
            DATASET MUST BE IN dataset_folder: '../data/Aerial/datasets/'
        """
        assert root in os.listdir(self.dataset_folder)
        assert split in ('train', 'test')
        self.root = self.dataset_folder + '/' + root
        with open('{}/boxes_{}.pkl'.format(self.root, split), 'rb') as handle:
            self.bounding_boxes = pickle.load(handle)       
        self.root += '/{}'.format(split)
        self.true_paths = os.listdir('{}/true'.format(self.root))
        self.transform = transform
        self.resolution = resolution

    def __len__(self):
        return len(self.bounding_boxes)

    def __getitem__(self, idx):
        uri = '{}/true/{}'.format(self.root, self.true_paths[idx])
        img = Image.open(uri).convert('RGB')
        boxes = self.bounding_boxes[uri]['boxes'] / self.resolution
        # show_bounding_boxes(img, boxes)
        if self.transform is not None:
            img = self.transform(img)
        return img, boxes
    
    def collate_fn(self, batch):
        images, boxes = list(), list()
        for item in batch:
            images.append(item[0])
            boxes.append(item[1])
        return torch.stack(images, dim=0), boxes


def localize_dataset(root, batch_size, batch_size_test, n_processes, resolution, transform=transforms.Compose([transforms.ToTensor()])):
    trainset = ObjectDetection(root=root, split='train', transform=transform, resolution=resolution)
    # trainset[86]
    testset = ObjectDetection(root=root, split='test', transform=transform, resolution=resolution)
    return DataLoader(
        dataset=trainset, collate_fn=trainset.collate_fn, batch_size=batch_size, 
            shuffle=True, num_workers=n_processes), \
                DataLoader(
                    dataset=testset, collate_fn=testset.collate_fn, batch_size=batch_size_test,
                         shuffle=False, num_workers=n_processes)
    
"""
root = 'train-10000_test-2000_res-256_lower-10_upper-24'
batch_size = 16
resolution = 256
batch_size_test = 4
n_threads = 1
transform = transforms.Compose([transforms.ToTensor()])
train, test = localize(root, batch_size, batch_size_test, transform, n_threads, resolution)"""
