import os
import cv2
import h5py
import torch
import pickle
import argparse 
import numpy as np 
from utils import *
from PIL import Image 
from glob import glob 
from copy import deepcopy
from imageio import imwrite 
from scipy.stats import truncnorm


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='../samples')
parser.add_argument('--target_h5', default='../data/generated_targets/train_data32x32.h5')
#parser.add_argument('--target_h5', default=('../data/generated_targets/train_data32x32.h5', '../data/generated_targets/test_data32x32.h5'), nargs='+')
# parser.add_argument('--background_folders', default=['../data/Aerial/backgrounds/sequoia256res', '../data/Aerial/backgrounds/MASATI-v1-low_res'], nargs='+')
parser.add_argument('--background_folders', default=['../data/Aerial/source_backgrounds/sequoia-rgb-images'], nargs='+')
parser.add_argument('--n', default=250, type=int)
parser.add_argument('--target_pixel_dist', default=(8, 55, 20, 12), nargs='+', type=int, \
    help='[0]: lower bound, [1]: upperbound, [2]: mean, [3]: stdev')
parser.add_argument('--n_targets', default=(10, 50, 30, 10), nargs='+', type=int, \
    help='[0]: lower bound, [1]: upperbound, [2]: mean, [3]: stdev')
parser.add_argument('--resolution', default=(3036, 4048, 3), nargs='+', type=int)
parser.add_argument('--resize_background', action='store_false')
parser.add_argument('--blend_range', default=(0.85, 1.0), nargs='+', type=float)

opt = parser.parse_args()
    

def draw_background(background_paths):
    other_paths = glob('../data/Aerial/source_backgrounds/Reduced_MASATI-v1/*.JPG')
    other_paths = [path for path in other_paths if ('multi' not in path and 'ship' not in path and 'detail' not in path)]
    np.random.shuffle(other_paths)
  
    background_paths = list(background_paths) + list(other_paths[:500])
    unused_paths = iter(list(other_paths[500:]))
    np.random.shuffle(background_paths)
    for path in background_paths:
        if 'MASATI' in path:
            img = np.asarray(Image.open(path).convert('RGB')) / 255
            while (img.shape[0] < 500 or img.shape[1] < 500):
                img = np.asarray(Image.open(unused_paths.__next__()).convert('RGB')) / 255
            yield img
        yield np.asarray(Image.open(path).convert('RGB')) / 255


assert [os.path.exists(folder) for folder in opt.background_folders]
assert os.path.exists(opt.save_dir)
assert os.path.exists(opt.target_h5) 

assert len(opt.resolution) == 3 and opt.resolution[-1] == 3

save_loc = '{save_dir}/samples-{n}.n_targets--{n_targets0}-{n_targets1}.pixels--{pix0}-{pix1}.res-{res0}x{res1}'.format(save_dir=opt.save_dir,\
     n=opt.n, n_targets0=opt.n_targets[0], n_targets1=opt.n_targets[1], pix0=opt.target_pixel_dist[0], pix1=opt.target_pixel_dist[1],\
        res0=opt.resolution[0], res1=opt.resolution[1])

background_paths = []
for folder in opt.background_folders:
    assert os.path.exists(folder)
    folder_files = glob('{}/*.JPG'.format(folder))
    assert len(folder_files) != 0
    background_paths += folder_files

background_paths = np.array(background_paths)
np.random.shuffle(background_paths)
n_backgrounds = len(background_paths)

backgrounds = draw_background(background_paths)


features = ['shape', 'shape_color', 'letter', 'letter_color', 'rotation']

n_min, n_max, mu, stdev = opt.n_targets
pxl_min, pxl_max, pxl_mu, pxl_stdev = opt.target_pixel_dist

def get_target_params():
    n = int(truncnorm((n_min - mu) / stdev,\
        (n_max - mu) / stdev, loc=mu, scale=stdev).rvs(1)[0])
    color_labels = np.random.choice(a=np.arange(0.01, 1.0, 0.01), size=n, replace=False)
    target_idxs = np.random.randint(low=0, high=n_available, size=n)
    resizes = np.asarray(
        truncnorm(
            (pxl_min - pxl_mu) / pxl_stdev, (pxl_max - pxl_mu) / pxl_stdev, \
                loc=pxl_mu, scale=pxl_stdev).rvs(n), dtype='uint8')
    for idx, color_label, size in zip(target_idxs, color_labels, resizes):
        yield cv2.resize(targets[idx], dsize=(size, size), interpolation=cv2.INTER_NEAREST), color_label, size, labels[:, idx]


os.mkdir(save_loc)
total_targets_placed = 0
h5_file = h5py.File(opt.target_h5, 'r+')
targets = np.asarray(h5_file['/X']) / 255

n_available = len(targets)
print('total backgrounds = {} | total samples = {} | num available targets = {}'.format(n_backgrounds, opt.n, n_available))
labels = np.asarray([np.asarray(h5_file["/{feature}".format(feature=feature)]) for feature in features])
save_template = '/{}.JPG'
boxes_dict = dict()
for save_idx in range(opt.n):
    bounding_boxes = []
    background = backgrounds.__next__()
    if background.shape != opt.resolution:
        background = cv2.resize(background, (opt.resolution[1], opt.resolution[0]),\
             interpolation=cv2.INTER_NEAREST)
    copy = deepcopy(background)
    instance_label = np.zeros(shape=opt.resolution)
    
    for idx, (tar, color_label, tar_size, features_i) in enumerate(get_target_params()):
        # tar = cv2.GaussianBlur(tar, ksize=(1, 1), sigmaX=1.3)
        mask = 0 ** np.ceil(tar)
        j1 = np.random.randint(0, opt.resolution[0] - tar_size, size=1)[0]
        i1 = np.random.randint(0, opt.resolution[1] - tar_size, size=1)[0]
        i2, j2 = i1 + tar_size, j1 + tar_size
        if idx <= 1:
            bounding_boxes.append([j1, j2, i1, i2])
            background[j1:j2, i1:i2, :] *= mask
            instance_label[j1:j2, i1:i2, :] = (0 ** mask) * color_label
            background[j1:j2, i1:i2, :] += tar
        elif idx >= 2 and check_collisions(bounding_boxes, [i1, i2, j1, j2]):
            bounding_boxes.append([j1, j2, i1, i2])
            background[j1:j2, i1:i2, :] *= mask
            instance_label[j1:j2, i1:i2, :] = (0 ** mask) * color_label
            background[j1:j2, i1:i2, :] += tar
    blend_factor = np.random.uniform(low=opt.blend_range[0], high=opt.blend_range[1])
    background = cv2.GaussianBlur(
        np.asarray((background * blend_factor + copy * (1 - blend_factor)) * 255, dtype='uint8'),\
             ksize=(3, 3), sigmaX=1)
    obj_ids = np.unique(instance_label)[1:]
    boxes = []
    for obj in obj_ids:
        pos = np.where(instance_label == obj)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    uri = save_loc + save_template.format(save_idx)
    imwrite(uri=uri, im=background)
    boxes_dict.update({uri: {'boxes': boxes, 'features': features_i}})
    total_targets_placed += len(bounding_boxes)
    if save_idx % 5 == 0:
        print('[{}/{}] samples generated\n{} targets placed'.format(save_idx, opt.n, total_targets_placed))
save_boxes(boxes_dict=boxes_dict, save_loc=save_loc)

