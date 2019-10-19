import os
import cv2
import h5py
import torch
import pickle
import argparse 
import numpy as np 
from PIL import Image 
from glob import glob 
from copy import deepcopy
from imageio import imwrite 

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='../data/Aerial/datasets')
parser.add_argument('--target_h5', default=('../data/generated_targets/train_data32x32.h5', \
    '../data/generated_targets/test_data32x32.h5'), nargs='+')
parser.add_argument('--background_folders', default=['../data/Aerial/backgrounds/sequoia256res', '../data/Aerial/backgrounds/MASATI-v1-low_res'], nargs='+')
parser.add_argument('--n_train', default=10000, type=int)
parser.add_argument('--n_test', default=2000, type=int)
parser.add_argument('--target_pixel_dist', default=(10, 24), nargs='+', type=int)
parser.add_argument('--altitude', default=0, type=int, help='if nonzero, it will be used to build target resolutions')
parser.add_argument('--n_targets', default=(0, 12), nargs='+', type=int)
parser.add_argument('--resolution', default=(300, 300, 3), nargs='+', type=int)
parser.add_argument('--resize_background', default=1, type=int)
parser.add_argument('--blur_factor', default=1.5, type=float)
parser.add_argument('--blend_range', default=(0.9, 1.0), nargs='+', type=float)

args = parser.parse_args()


def check_collisions(bounding_boxes, new_bounds):
    i1, i2, j1, j2 = new_bounds
    i, j = np.mean([i1, i2]), np.mean([j1, j2])
    dist = abs(i1 - i2)
    for b in bounding_boxes:
        m1, m2 = np.mean([b[0], b[1]]), np.mean([b[2], b[3]])
        if np.sqrt(np.power(i - m1, 2) + np.power(j - m2, 2)) < 2 * dist:
            return False
    return True

def save_boxes(boxes_dict, split):
    pickle_file = '{}/boxes_{}.pkl'.format(save_loc, split)
    with open(pickle_file, 'wb') as handle:
        pickle.dump(boxes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved bounding boxes to {}'.format(pickle_file))

assert os.path.exists(args.save_dir)
assert len(args.target_h5) == 2 and os.path.exists(args.target_h5[0]) and os.path.exists(args.target_h5[1])
assert [os.path.exists(folder) for folder in args.background_folders]

assert len(args.resolution) == 3 and args.resolution[-1] == 3
assert args.n_train > args.n_test

save_loc = '{save_dir}/train-{n_train}_test-{n_test}_res-{res}_lower-{lower}_upper-{upper}'.format(save_dir=args.save_dir, \
    n_train=args.n_train, n_test=args.n_test, res=args.resolution[0], lower=args.target_pixel_dist[0], upper=args.target_pixel_dist[1])

assert not os.path.exists(save_loc)

# check valid background paths and store into array
background_paths = []
for folder in args.background_folders:
    assert os.path.exists(folder)
    folder_files = glob('{}/*.JPG'.format(folder))
    assert len(folder_files) != 0
    background_paths += folder_files

background_paths = np.array(background_paths)
np.random.shuffle(background_paths)

split_ratio = args.n_test / (args.n_test + args.n_train)
n_backgrounds = len(background_paths)
n_train_backgrounds = int((1 - split_ratio) * len(background_paths))
n_test_backgrounds = n_backgrounds - n_train_backgrounds
train_backgrounds = background_paths[:n_train_backgrounds]
test_backgrounds = background_paths[n_train_backgrounds:]
print('total backgrounds: {} = ({} train + {} test)'.format(n_backgrounds, n_train_backgrounds, n_test_backgrounds))

target_h5 = {'train': args.target_h5[0], 'test': args.target_h5[1]}
features = ['shape', 'shape_color', 'letter', 'letter_color', 'rotation']

size_lower, size_upper = args.target_pixel_dist
n_min, n_max = args.n_targets


def build_new(split, features_i, save_idx):
    bounding_boxes = []
    n_targets = np.random.randint(low=n_min, high=n_max, size=1)[0]
    target_idxs = np.random.randint(low=0, high=available_targets, size=n_targets)
    drawn_targets = iter(targets[np.random.randint(low=0, high=available_targets, size=n_targets)])
    rand_background_idx = np.random.randint(low=0, high=n_train_backgrounds if split=='train' else n_test_backgrounds)
    if split == 'train':
        background = np.asarray(Image.open(train_backgrounds[rand_background_idx]).convert('RGB')) / 255
    else: 
        background = np.asarray(Image.open(test_backgrounds[rand_background_idx]).convert('RGB')) / 255
    assert background.shape == args.resolution
    copy = deepcopy(background)
    instance_label = np.zeros(shape=args.resolution)
    color_labels = iter(np.random.choice(a=np.arange(0.1, 1.0, 0.02), size=n_targets, replace=False))
    for i in range(n_targets):
        resize_dim = np.random.randint(low=size_lower, high=size_upper)
        src = cv2.resize(drawn_targets.__next__(), dsize=(resize_dim, resize_dim), interpolation=cv2.INTER_NEAREST)
        src = cv2.GaussianBlur(src, ksize=(1, 1), sigmaX=args.blur_factor)
        mask = 0 ** np.ceil(src)
        i1, j1 = np.random.randint(resize_dim, args.resolution[0] - resize_dim, size=2)
        i2, j2 = i1 + resize_dim, j1 + resize_dim
        if i <= 1:
            bounding_boxes.append([i1, i2, j1, j2])
            background[i1:i2, j1:j2, :] *= mask
            instance_label[i1:i2, j1:j2, :] = (0 ** mask) * next(color_labels)
            background[i1:i2, j1:j2, :] += src
        elif i >= 2 and check_collisions(bounding_boxes, [i1, i2, j1, j2]):
            bounding_boxes.append([i1, i2, j1, j2])
            background[i1:i2, j1:j2, :] *= mask
            instance_label[i1:i2, j1:j2, :] = (0 ** mask) * next(color_labels)
            background[i1:i2, j1:j2, :] += src
    blend_factor = np.random.uniform(low=args.blend_range[0], high=args.blend_range[1])
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
    uri = template.format('true', save_idx)
    imwrite(uri=template.format('true', save_idx), im=background)
    if save_idx == 0: 
        print(uri)

    boxes_dict.update({uri: {'boxes': boxes, 'features': features_i}})
    return len(bounding_boxes)
    

os.mkdir(save_loc)
for split in ('train', 'test'):
    total_targets = 0
    for new_dir in ('/{}', '/{}/true'):
        os.mkdir(save_loc + new_dir.format(split))
    h5_file = h5py.File(target_h5[split], 'r+')
    targets = np.asarray(h5_file['/X']) / 255
    available_targets = len(targets)
    labels = np.asarray([np.asarray(h5_file["/{feature}".format(feature=feature)]) for feature in features])
    template = save_loc + '/{}'.format(split) + '/{}/{}.JPG'
    boxes_dict = dict()
    n_iters = args.n_train if split == 'train' else args.n_test
    for i in range(n_iters):
        total_targets += build_new(split, labels[:, i], i)
        if i % 50 == 0:
            print('({}) total targets used = {} | images created = {}'.format(split, total_targets, i))
    save_boxes(boxes_dict, split)


