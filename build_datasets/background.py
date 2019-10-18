import os
import argparse 
import numpy as np
from PIL import Image
from glob import glob
from imageio import imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='../data/Aerial/')
parser.add_argument('--input_folders', nargs='+', required=True, type=str)
parser.add_argument('--resolution', default=256, type=int)
parser.add_argument('--name', required=True, type=str)

args = parser.parse_args()

save_loc = args.save_dir + args.name + str(args.resolution)

assert os.path.exists(args.save_dir)
assert not os.path.exists(save_loc)

background_paths = []
for folder in args.input_folders:
    assert os.path.exists(folder)
    folder_files = glob('{}/*.JPG'.format(folder))
    assert len(folder_files) != 0
    background_paths += folder_files
np.random.shuffle(background_paths)

os.mkdir(save_loc)
for idx, file in enumerate(background_paths):
    img = np.asarray(Image.open(file).convert('RGB'))
    h, w = img.shape[:-1]
    if not (h < args.res or w < args.res):
        count = 0
        i_iter = h // args.res
        i = 0
        while i_iter != 0:
            j = 0
            j_iter = w // args.res
            while j_iter != 0:
                imwrite(uri='{save_loc}/{idx}-{i}-{j}.JPG'.format(save_loc=save_loc, idx=idx, i=i_iter, j=j_iter), \
                    im=img[i: i+args.res, j: j+args.res])
                count += 1
                j += args.res
                j_iter -= 1
            i_iter -= 1
            i += args.res

    if idx % 5 == 0:
        print('count: {} | iteration {}'.format(count, idx))
