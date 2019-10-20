import os
import argparse 
import numpy as np
from PIL import Image
from glob import glob
from imageio import imwrite

src_backgrounds = os.listdir('../data/Aerial/source_backgrounds')

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='../data/Aerial/backgrounds/')
parser.add_argument('--input_folders', nargs='+', type=str, default=src_backgrounds)
parser.add_argument('--resolution', default=256, type=int)
parser.add_argument('--name', required=True, type=str)

args = parser.parse_args()
input_folders = ['../data/Aerial/source_backgrounds/' + folder for folder in args.input_folders if folder != '.DS_Store']
print(input_folders)
print([os.path.exists(path) for path in input_folders])

save_loc = args.save_dir + args.name + '.res-{}'.format(args.resolution)

assert os.path.exists(args.save_dir)
assert not os.path.exists(save_loc)

background_paths = []
for folder in input_folders:
    assert os.path.exists(folder)
    folder_files = glob('{}/*.JPG'.format(folder))
    assert len(folder_files) != 0
    background_paths += folder_files
np.random.shuffle(background_paths)

os.mkdir(save_loc)
total_collected = 0
res = args.resolution
for idx, file in enumerate(background_paths):
    img = np.asarray(Image.open(file).convert('RGB'))
    h, w = img.shape[:-1]
    if not (h < res or w < res):
        i_iter = h // res
        i = 0
        while i_iter != 0:
            j = 0
            j_iter = w // res
            while j_iter != 0:
                imwrite(uri='{save_loc}/{idx}-{i}-{j}.JPG'.format(save_loc=save_loc, idx=idx, i=i_iter, j=j_iter), \
                    im=img[i: i+res, j: j+res])
                total_collected += 1
                j += res
                j_iter -= 1
            i_iter -= 1
            i += res

    if idx % 5 == 0:
        print('count: {} | iteration {}'.format(total_collected, idx))
