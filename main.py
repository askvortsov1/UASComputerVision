'''
COMPUTER VISION PIPELINE FOR AUVSI-SUAS 2020 COMPETITION
'''

import os
import time
import json
import shutil
import requests
import argparse

import torch
import numpy as np
from PIL import Image

import colored


parser = argparse.ArgumentParser()
parser.add_argument('--IP', default='', type=str)
parser.add_argument('--scheme', default='http://', type=str)
parser.add_argument('--port', default='8000', type=str)
parser.add_argument('--get_path', default='api/getimage/', type=str)
parser.add_argument('--post_path', default='api/senddata/', type=str)
parser.add_argument('--resolution', default=(4048, 3036, 3), nargs='+', type=int)

parser.add_argument('--server', action='store_true')
parser.add_argument('--detector_path', default='', type=str)
parser.add_argument('--resize', default=300, type=int)
parser.add_argument('--sample_dir', default='', type=str)


opt = parser.parse_args()

# features = {'shape', 'shape_color', 'letter', 'letter_color', 'rotation'}


def GETPOST_URL_BUILD():
    get_url = '{scheme}{IP}:{port}/{path}'.format(scheme=opt.scheme, IP=opt.IP, \
        port=opt.port, path=opt.get_path)
    post_url = '{scheme}{IP}:{port}/{path}'.format(scheme=opt.scheme, IP=opt.IP, \
        port=opt.port, path=opt.post_path)
    return get_url, post_url 


def process(image):

    return None


if __name__ == '__main__':
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
        for image_name in os.listdir(opt.sample_dir):
            if image_name[-3:].upper() in {'PNG', 'JPG'}:
                image = Image.open('{dir}/{name}'.format(dir=opt.sample_dir, name=image_name))
                target_data = process(image)
                for idx, (pixel_coord, features) in enumerate(target_data.items()):
                    print('target #{}: '.format(idx + 1))
                    print('[{loc}] -> shape: {features.shape}, shape-color: {features.sh_col},' + \
                        ' letter: {features.letter}, letter-color: {features.l_col},' + \
                            ' orientation: {features.rot}\n\n'.format(pixel_coord, features))

            








