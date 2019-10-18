import os
import cv2
import h5py
import pickle
import argparse 
import numpy as np
import pandas as pd 
from PIL import Image
from glob import glob
from copy import deepcopy
from imageio import imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test')
parser.add_argument('--save_loc', default='../data/generated_targets/low_res_aerial')
parser.add_argument('--h5_file', default='../data/generated_targets/test_data32x32.h5')
parser.add_argument('--bg_h5', default='../data/Aerial/seq256.h5')
parser.add_argument('--n_build', default=4000, type=int)
parser.add_argument('--target_range', default=(4, 16), nargs='+', type=int)
parser.add_argument('--n_targets_range', default=(0, 12), nargs='+', type=int)
parser.add_argument('--build_bgs', default=1, type=int)

args = parser.parse_args()
BG_MASATI = '../data/Aerial/Reduced_MASATI-v1'
BG_SEQUOIA = '../data/Aerial/sequoia-rgb-images'
SEQUOIA256RES = '../data/Aerial/sequoia256res'
MLOW_RES = '../data/Aerial/MASATI-v1-low_res'


I, J = 3036 // 256, 4048 // 256
I += 1
J += 1
print(J, 4048 / 256)
import torch
import torchvision 
import matplotlib.pyplot as plt
files = glob('../data/generated_targets/simulated_fields_low_res/train/true/*.JPG')
np.random.shuffle(files)
use_files = files[:I * J]
imgs = torch.Tensor([np.transpose(np.asarray(Image.open(path).convert('RGB'))/255, (2, 0, 1)) for path in use_files])
print(imgs.size())
X = np.transpose(torchvision.utils.make_grid(imgs, padding=0, nrow=J).numpy(), (1, 2, 0))
 
fig = plt.gcf()
fig.set_size_inches(50, 50)
plt.imshow(X)
plt.savefig('/Users/lukeottey/anaconda3/envs/uas_repo/UASComputerVision/perspective_from_heli.JPG')
plt.show()
print(X.shape)

quit()
if args.build_bgs:
    res = 256
    file_paths = glob('{}/*/*.png'.format(BG_MASATI))
    np.random.shuffle(file_paths)
    for idx, file in enumerate(file_paths):
        img = np.asarray(Image.open(file).convert('RGB'))
        if not (img.shape[0] < 256 or img.shape[1] < 256):
            count = 0
            i_iter = img.shape[0] // res
            i = 0
            while i_iter != 0:
                j = 0
                j_iter = img.shape[1] // res
                while j_iter != 0:
                    imwrite(uri='{dir}/masati-v1{res}-{idx}-{i}-{j}.JPG'.format(dir=MLOW_RES, res=res, idx=idx, i=i_iter, j=j_iter), \
                        im=img[i: i+res, j: j+res])
                    count += 1
                    j += res
                    j_iter -= 1
                i_iter -= 1
                i += res
    print('count: {}'.format(count))
    if idx % 5 == 0:
        print('iteration {}'.format(idx))
    
    quit()

"""
if args.build_bgs:
    ""
    BUILD REDUCED SEQUOIA DATASET OF RESOLUTION 256
    "
    data = []
    file_paths = glob('{}/*.JPG'.format(BG_SEQUOIA))
    np.random.shuffle(file_paths)
    for idx, file in enumerate(file_paths):
        img = np.asarray(Image.open(file).convert('RGB'))
        res = np.random.randint(low=800, high=1000, size=1)[0]
        count = 0
        for i in range(0, img.shape[0], res):
            for j in range(0, img.shape[1], res):
                
                imwrite(uri='{dir}/seq{res}-{idx}-{i}-{j}.JPG'.format(dir=SEQUOIA256RES, res=res, idx=idx, i=i, j=j), \
                    im=cv2.resize(img[i: i+res, j: j+res], (256, 256), cv2.INTER_NEAREST))
                count += 1
        print('RESOLUTION: {} | count: {}'.format(res, count))
        if idx % 5 == 0:
            print('iteration {}'.format(idx))"""

assert [os.path.exists(path) for path in (args.save_loc, args.h5_file, args.bg_h5, BG_MASATI)]

background_paths = glob('{}/*.JPG'.format(SEQUOIA256RES))
np.random.shuffle(background_paths)


class ScatterTargets:
    def __init__(self, folder, targets, bg_paths):
        file = h5py.File(targets, "r+")
        self.targets = np.asarray(file["/X"]) / 255
        features = ['shape', 'shape_color', 'letter', 'letter_color', 'rotation']
        self.labels = np.array([np.asarray(file["/{feature}".format(feature=feature)]) for feature in features])
        self.boxes_dict = dict()
        self.bg_paths = bg_paths
        self.n_bgs = len(bg_paths)
        self.total_targets = 0
        self.size_l, self.size_u = args.target_range 
        self.n_min, self.n_max = args.n_targets_range
        self.n_calls = 0

    def random_background(self):
        idx = np.random.randint(low=0, high=self.n_bgs)
        
        return np.asarray(Image.open(self.bg_paths[idx]).convert('RGB')) / 255

    def check_collisions(self, bounding_boxes, new_bounds):
        i1, i2, j1, j2 = new_bounds
        i, j = np.mean([i1, i2]), np.mean([j1, j2])
        dist = abs(i1 - i2)
        for b in bounding_boxes:
            m1, m2 = np.mean([b[0], b[1]]), np.mean([b[2], b[3]])
            if np.sqrt(np.power(i - m1, 2) + np.power(j - m2, 2)) < 2 * dist:
                return False
        return True

    def __folder__(self):
        return len(os.listdir('../data/generated_targets/simulated_fields_low_res/{}/true'.format(args.split)))

    def __num_boxes__(self):
        print('{} total objects in {} images'.format(self.total_targets, self.__folder__()))

    def __save__(self):
        with open('../data/generated_targets/simulated_fields_low_res/boxes_{}.pkl'.format(args.split), 'wb')\
                as handle:
            pickle.dump(self.boxes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self):
        self.n_calls += 1
        bounding_boxes = []
        n = np.random.randint(low=self.n_min, high=self.n_max, size=1)
        idxs = np.random.randint(low=0, high=len(self.targets), size=n)
        BG = self.random_background()
        BG_copy = deepcopy(BG)
        imgs = iter(self.targets[idxs])
        #assert len(imgs) == len(idxs)
        segmented = np.zeros(shape=(256, 256, 3))
        colors = iter(np.random.choice(a=np.arange(0.1, 1.0, 0.02), size=n[0], replace=False))
        for i in range(n[0]):
            resize_dim = np.random.randint(low=self.size_l, high=self.size_u)
            src = cv2.resize(imgs.__next__(), dsize=(resize_dim, resize_dim), interpolation=cv2.INTER_NEAREST)
            src = cv2.GaussianBlur(src, ksize=(1, 1), sigmaX=1.5)
            mask = 0 ** np.ceil(src)
            i1 = np.random.randint(resize_dim, BG.shape[0] - resize_dim)
            i2 = i1 + resize_dim
            j1 = np.random.randint(resize_dim, BG.shape[1] - resize_dim)
            j2 = j1 + resize_dim
            if i <= 1:
                bounding_boxes.append([i1, i2, j1, j2])
                BG[i1:i2, j1:j2, :] *= mask
                segmented[i1:i2, j1:j2, :] = (0 ** mask) * next(colors)
                src_z = np.zeros_like(BG)
                src_z[i1:i2, j1:j2, :] = src
                BG = src_z + BG
            elif i > 1 and self.check_collisions(bounding_boxes, [i1, i2, j1, j2]):
                bounding_boxes.append([i1, i2, j1, j2])
                BG[i1:i2, j1:j2, :] *= mask
                segmented[i1:i2, j1:j2, :] = (0 ** mask) * next(colors)
                src_z = np.zeros_like(BG)
                src_z[i1:i2, j1:j2, :] = src
                BG = src_z + BG
        blend_ratio = np.random.uniform(low=0.90, high=1.0)
        uri = template.format('mask', self.__folder__())
        imwrite(uri=uri, im=np.array(segmented * 255, dtype='uint8'))
        BG = np.array((BG * blend_ratio + BG_copy * (1 - blend_ratio)) * 255, dtype='uint8')
        BG = cv2.GaussianBlur(BG, ksize=(3, 3), sigmaX=1)
        uri = template.format('true', self.__folder__())
        self.boxes_dict.update({uri: bounding_boxes})
        self.total_targets += len(bounding_boxes)
        imwrite(uri=uri, im=BG)
        if self.n_calls % 50 == 0:
            self.__num_boxes__()


template = '../data/generated_targets/simulated_fields_low_res/{}'.format(args.split) + '/{}/{}.JPG'
s = ScatterTargets('_', args.h5_file, background_paths)
for i in range(args.n_build):
    s()
s.__save__()








