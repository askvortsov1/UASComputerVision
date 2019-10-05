import os
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from utils.transformations import FractionalMaxPool, AddBackground


__all__ = ['localize']


root = '/Users/lukeottey/anaconda3/envs/uas2/Vision/data/generated_targets/simulated_fields/train'
img = Image.open('{}/true/{}'.format(root, '357.png')).convert('RGB')
mask = np.array(Image.open('{}/mask/{}'.format(root, '357.png')))
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

"""
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
quit()"""
import torchvision, cv2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

n_boxes = len(boxes)
fig, ax = plt.subplots(ncols=1, nrows=2)
fig.set_size_inches(20, 20)
#ax[1].imshow(mask)
#ax[2].imshow(img)"""
bxs = []
img = np.array(img)
for i, box in enumerate(boxes):
    box = box.numpy()
    bxs.append(img[int(box[1]): int(box[3]), int(box[0]): int(box[2])])
    #rect = patches.Rectangle((int(box[0]), box[1]), int(box[2] - box[0]), int(box[3] - box[1]), linewidth=0.75, edgecolor='r', facecolor='none')
    #ax[2].add_patch(rect)

for i in range(n_boxes):
    bxs[i] = np.transpose(cv2.resize(bxs[i]/255, dsize=(64, 64), interpolation=cv2.INTER_NEAREST), (2, 0, 1))
    print(bxs[i].shape)
img = Image.open('/Users/lukeottey/anaconda3/envs/uas2/Vision/data/generated_targets/60000/uas_localization_ex4 copy.png').convert('RGB')
ax[0].imshow(img)
ax[0].axis('off')
bxs = torch.Tensor(bxs)
print(bxs.size())
ax[1].imshow(np.transpose(torchvision.utils.make_grid(bxs, nrow=7, padding=0).numpy(), (1, 2, 0)))
plt.show()
quit()
"""
fig, ax = plt.subplots(ncols=n_boxes//2, nrows=2)
fig.set_size_inches(20, 20)
for i, a in enumerate(ax, 0):
    a.imshow(bxs[i]/255, interpolation='none')
plt.show()
quit()"""



class ObjectDetection(Dataset):
    def __init__(self, root='./data/generated_targets/simulated_fields', split='train', transform=None):
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


def localize(args, train, test):
    return_arr = []
    if train:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        trainset = ObjectDetection(transform=train_transforms, split='train')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        return_arr.append(trainloader)
    if test:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = ObjectDetection(transform=test_transforms, split='test')
        testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
        return_arr.append(testloader)
    if len(return_arr) == 1:
        return return_arr[0]
    return return_arr    




