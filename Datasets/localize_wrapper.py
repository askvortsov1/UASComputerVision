import os
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformations import AddBG, FractionalMaxPool

__all__ = ['localize']


class ObjectDetection(Dataset):
    def __init__(self, root='../../data/generated_targets/simulated_fields', split='train', transform=None):
        assert split in ('train', 'test')
        self.root = '{}/{}'.format(root, split)
        with open('{}/boxes_{}.pkl'.format(root, split), 'rb') as handle:
            self.bounding_boxes = pickle.load(handle)
        self.true_paths = os.listdir('{}/true'.format(self.root))
        self.mask_paths = os.listdir('{}/mask'.format(self.root))
        self.transform = transform

    def __len__(self):
        return len(self.bounding_boxes)

    def __getitem__(self, idx):
        img = Image.open('{}/true/{}'.format(self.root, self.true_paths[idx])).convert('RGB')
        mask = np.array(Image.open('{}/mask/{}'.format(self.root, self.mask_paths[idx])))
        obj_ids = np.unique(mask)[1:]
        boxes = []
        for obj in obj_ids:
            pos = np.where(mask == obj)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        show = 1
        if show:
            import matplotlib.pyplot as plt 
            import matplotlib.patches as patches 
            fig, ax = plt.subplots(ncols=3)
            fig.set_size_inches(20, 20)
            ax[0].imshow(img)
            ax[1].imshow(mask)
            ax[2].imshow(img)
            for box in boxes:
                box = box.numpy()
                rect = patches.Rectangle((int(box[0]), box[1]), int(box[2] - box[0]), int(box[3] - box[1]), linewidth=0.75, edgecolor='r', facecolor='none')
                ax[2].add_patch(rect)
            plt.show()
            quit()
        if self.transform is not None:
            img = self.transform(img)
        return img, {'boxes':boxes, 'areas': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])}#target #{'boxes': boxes, 'areas': areas}


def localize(args):
    # TODO FILL OUT THIS FUNCTION, RETURN TRAIN AND TEST DATALOADERS
    pass