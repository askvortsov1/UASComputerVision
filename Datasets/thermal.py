import os
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


__all__ = ['thermal']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mask_path = '/Users/lukeottey/anaconda3/envs/uas2/Vision/data/generated_targets/simulated_fields/train/mask/2.png'
    true_path = '/Users/lukeottey/anaconda3/envs/uas2/Vision/data/generated_targets/simulated_fields/train/true/2.png'
    img = np.array(Image.open(mask_path).convert('RGB')) / 255
    img = np.ceil(img)
    plt.imshow(img)
    plt.show()    








