import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from copy import deepcopy
from imageio import imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', default='bounding_boxes.csv', type=str,
                    help='all CSVs will be saved to directory ./data/CSV')
parser.add_argument('--train', default=0, type=int)
parser.add_argument('--img_dir', default='scattered_targets', type=str,
                    help='all image directories will be saved to directory ./data/generated_targets')
parser.add_argument('--input_csv', default='./data/CSV/test10000_csv.csv')
parser.add_argument('--n_imgs', default=100, type=int)
parser.add_argument('--bgdir_path', default='./data/Aerial/sequoia-rgb-images', type=str)
parser.add_argument('--targets_dir', default='./data/generated_targets/run1', type=str)
parser.add_argument('--resize_min', default=10, type=int)
parser.add_argument('--resize_max', default=150, type=int)
parser.add_argument('--lower', default=2, type=int, help='min number of targets to be placed within the image')
parser.add_argument('--upper', default=25, type=int, help='max number of targets to be placed within the image')


class ScatterTargets:
    def __init__(self, dataframe, folder):
        self.dataframe = dataframe
        self.folder = folder
        self.boxes_dict = dict()
        bg_paths = np.concatenate((glob(args.bgdir_path + '/*'), glob('./data/Aerial/MASATI-v1/*/*.png'))).tolist()
        self.random_bg = lambda: np.array(Image.open(np.random.choice(bg_paths)).convert('RGB')) / 255.
        self.paths = lambda n, idxs: np.random.choice(self.dataframe['path'].values[idxs], replace=False, size=n)
        self.open = lambda n, idxs: iter([np.array(
            Image.open(x).convert('RGB')) / 255.
                                          for x in np.random.choice(
                self.dataframe['path'].values[idxs], replace=False, size=n)])
        self.total_targets = 0

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
        return len(os.listdir(self.folder))

    def __num_boxes__(self):
        print('{} total objects in {} images'.format(self.total_targets, self.__folder__()))

    def __save__(self, train=True):
        with open('./data/generated_targets/simulated_fields/boxes_{}.pkl'.format('train' if train else 'test'), 'wb')\
                as handle:
            pickle.dump(self.boxes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self):
        bounding_boxes = []
        n = np.random.randint(low=args.lower, high=args.upper, size=1)
        idxs = np.random.randint(low=0, high=len(self.dataframe), size=n)
        BG = self.random_bg()
        BG = cv2.resize(BG, (2048, 2048), interpolation=cv2.INTER_NEAREST)
        BG_copy = deepcopy(BG)
        imgs = self.open(n, idxs)
        segmented = np.zeros(shape=(2048, 2048, 3))
        colors = iter(np.random.choice(a=np.arange(0.1, 1.0, 0.02), size=n[0], replace=False))
        for i in range(n[0]):
            resize_dim = np.random.randint(low=args.resize_min, high=args.resize_max)
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
        blend_ratio = np.random.uniform(low=0.85, high=1.0)
        uri = template.format('mask', self.__folder__())
        imwrite(uri=uri, im=np.array(segmented * 255, dtype='uint8'))
        BG = np.array((BG * blend_ratio + BG_copy * (1 - blend_ratio)) * 255, dtype='uint8')
        BG = cv2.GaussianBlur(BG, ksize=(3, 3), sigmaX=1)
        uri = template.format('true', self.__folder__())
        self.boxes_dict.update({uri: bounding_boxes})
        self.total_targets += len(bounding_boxes)
        imwrite(uri=uri, im=BG)
        self.__num_boxes__()


if __name__ == '__main__':

    args = parser.parse_args()
    split = 'train' if args.train else 'test'
    folder = './data/generated_targets/simulated_fields/{}/true'.format(split)
    template = './data/generated_targets/simulated_fields/{}'.format(split) + '/{}/{}.png'

    df = pd.read_csv(args.input_csv)
    st = ScatterTargets(df, folder)
    for i in range(args.n_imgs):
        st()
    st.__save__(train=args.train)
    