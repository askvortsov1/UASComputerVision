import sys
sys.path.append("..")
import os
import argparse
import numpy as np
import pandas as pd
from utils import image_generation
from PIL import Image, ImageDraw, ImageOps


def check_dirs(args):
    if os.path.exists(args.csv_path):
        raise Exception('the CSV path {} already exits.'.format(args.csv_path))
    if os.path.exists(args.img_dir):
        raise Exception('the image directory {} already exits.'.format(args.img_dir))
    else:
        os.mkdir(args.img_dir)


def draw(sample, res, args):
    img = Image.new('RGBA', size=args.size[:-1], color=(0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    colorcode_shp = image_generation.HSLConversion(color=sample['shape_color'])
    image_generation.draw_shape(shape=sample['shape'], draw=dr, res=res, color=colorcode_shp)
    colorcode_let = image_generation.HSLConversion(sample['letter_color'])
    image_generation.draw_text(sample=sample, draw=dr, res=res, colorcode=colorcode_let)
    img = img.rotate(sample['rotation'])
    img = ImageOps.expand(img, border=int(res * 0.5), fill=(0))
    crop_val = int((img.width - args.size[0]) / 2)
    crop_box = (crop_val, crop_val, img.width - crop_val, img.height - crop_val)
    img = img.crop(crop_box)
    img.save(sample['path'])


def main(args):
    args.csv_path = './data/CSV/' + args.csv_path
    args.img_dir = './data/generated_targets/' + args.img_dir + '/train'
    if not os.path.exists(args.img_dir):
        os.mkdir(args.img_dir)
    path_template = os.getcwd()[:-8] + args.img_dir[3:] + '/{}.png'
    assert args.n > 0 and len(args.size) == 3 and args.size[-1] == 3
    check_dirs(args)
    sampler = image_generation.ATTR()
    samples = pd.DataFrame(vars(sampler.random_features(n=args.n, return_index=True)))
    samples['path'] = np.vectorize(lambda i: path_template.format(i))(np.arange(args.n))
    samples.to_csv(args.csv_path)
    for i, row in samples.iterrows():
        res = np.random.randint(args.lb_size, args.ub_size)
        draw(row, res, args)
        if i % 100 == 0:
            print('iteration: {}'.format(i))








