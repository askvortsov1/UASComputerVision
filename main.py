'''
COMPUTER VISION PIPELINE FOR AUVSI-SUAS 2020 COMPETITION
'''

import os
import cv2
import time
import json
import shutil
import requests
import argparse

import torch
import numpy as np
from PIL import Image
from itertools import product
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import colored


parser = argparse.ArgumentParser()
parser.add_argument('--IP', default='', type=str)
parser.add_argument('--scheme', default='http://', type=str)
parser.add_argument('--port', default='8000', type=str)
parser.add_argument('--get_path', default='api/getimage/', type=str)
parser.add_argument('--post_path', default='api/senddata/', type=str)
parser.add_argument('--server', action='store_true')

parser.add_argument('--crop_to_fit', action='store_true')
parser.add_argument('--cross_sections', default=4, type=int)
parser.add_argument('--resolution', default=(3036, 4048, 3), nargs='+', type=int)

parser.add_argument('--detector_path', default='', type=str)
parser.add_argument('--resize', default=300, type=int)
parser.add_argument('--sample_dir', default='../samples/samples-250.n_targets--10-50.pixels--8-55.res-3036x4048', type=str)
parser.add_argument('--subset_size', default=5, type=int, help='set to -1 to use full directory')
parser.add_argument('--crop_method', default='sixteen', choices=['sixteen', 'four'])

# features = {'shape', 'shape_color', 'letter', 'letter_color', 'rotation'}


def GETPOST_URL_BUILD():
    get_url = '{scheme}{IP}:{port}/{path}'.format(scheme=opt.scheme, IP=opt.IP, \
        port=opt.port, path=opt.get_path)
    post_url = '{scheme}{IP}:{port}/{path}'.format(scheme=opt.scheme, IP=opt.IP, \
        port=opt.port, path=opt.post_path)
    return get_url, post_url 


class CreateBatch(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = np.empty((16, opt.resize, opt.resize, 3), dtype='uint8')
        x = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                self.data[x] = np.asarray(data[i][j])
                x += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return transforms.ToTensor()(self.data[idx])


def four_crop(img):
    # (4048, 3036) -> 4 x (2024, 1518)
    img_width, img_height = img.size 
    output_size = (img_height // 2, img_width // 2)
    crop_height, crop_width = output_size
    if crop_width > img_width or crop_height > img_height:
        raise ValueError('Crop size larger than input size')
    return [img.crop((0, 0, crop_width, crop_height)),\
         img.crop((img_width - crop_width, 0, img_width, crop_height)),\
              img.crop((0, img_height - crop_height, crop_width, img_height)),\
                  img.crop((img_width - crop_width, img_height - crop_height, img_width, img_height))]

def sixteen_crop(img, **kwargs):
    # (4048, 3036) -> 16 x (1012, 759)
    if 'resize' in kwargs and kwargs['resize'] == True:
        assert 'output_size' in kwargs.keys()
        crops = four_crop(img)
        for i, crop in enumerate(crops):
            crops[i] = [resize(c, kwargs['output_size']) for c in four_crop(crop)]
        return crops
    else:
        tl, tr, bl, br = four_crop(img)
    return [np.asarray(four_crop(tl)), np.asarray(four_crop(tr)), np.asarray(four_crop(bl)), np.asarray(four_crop(br))]


def show_cropped_img(imgs, nrows, ncols):
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].imshow(imgs[i][j])
    plt.show()


def resize(img, output_size, interpolation=Image.BILINEAR):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return img.resize(output_size[::-1], interpolation)

    
def transform_original(img):
    img = sixteen_crop(img, resize=True, output_size=opt.resize)
    return img


def find_targets(img):
    # convert individual to tensor or create a batch of tensors
    

    return [0, 0, 1, 1]


def find_features(img, coords):
    return [None, None, None, None, None]


def bbox_to_xy(bbox):
    return [0, 0]


def process(image):
    image = transform_original(image)
    dataloader = DataLoader(dataset=CreateBatch(image), shuffle=False, batch_size=16)
    for x in dataloader:
        break
    del dataloader
    
    
    quit()
    I, J = len(image), len(image[0])
    for i in range(I):
        for j in range(J):
            bbox = find_targets(image[i][j])
            xy_coords = bbox_to_xy(bbox)
            #yield xy_coords, find_features(image[i][j], coords=bbox)


if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.server: 
        GET_URL, POST_URL = GETPOST_URL_BUILD()
        while True:
            r = requests.get(GET_URL, stream=True)
            if 'imageid' not in r.headers:
                print(colored.stylize('No images in queue.', colored.fg('red')))
                time.sleep(1)
                continue 
            imageid = r.headers['imageid']
            print('Processing image: {}'.format(imageid))
            if r.status_code == 200:
                with open('img.jpg', 'wb') as f:
                    r.raw.decode_content = True 
                    shutil.copyfileobj(r.raw, f)

            image = np.array(Image.open('img.jpg').convert('RGB'))
            target_data = process(image)
            json_data = {
                'imageid': imageid,
                'targets': target_data
            }
            print('Completed Process for Image: {}'.format(imageid))
            r = requests.post(POST_URL, stream=True, json=json_data)
    else:
        assert os.path.exists(opt.sample_dir)
        if opt.subset_size > 0: 
            subset = np.random.choice(os.listdir(opt.sample_dir), replace=False, size=opt.subset_size)
        else:
            subset = os.listdir(opt.sample_dir)
        for image_name in subset:
            if image_name[-3:].upper() in {'PNG', 'JPG'}:
                image = Image.open('{dir}/{name}'.format(dir=opt.sample_dir, name=image_name))
                target_data = process(image)
                for idx, (pixel_coord, features) in enumerate(target_data.items()):
                    print('target #{}: '.format(idx + 1))
                    print('[{loc}] -> shape: {features.shape}, shape-color: {features.sh_col},' + \
                        ' letter: {features.letter}, letter-color: {features.l_col},' + \
                            ' orientation: {features.rot}\n\n'.format(pixel_coord, features))

            








