import cv2
import random
import numpy as np
from glob import glob
from PIL import Image
from imageio import imread

class AddBG(object):
    def __init__(self, bg_prob, min_resize_ratio):
        self.__setattr__('probability', bg_prob)
        self.__setattr__('min_resize_ratio', min_resize_ratio)
        self.__setattr__('available_bg', glob('../../data/Aerial/MASATI-v1/*/*.png'))
        
    def random_background(self, resize):
        return cv2.resize(src=np.asarray(Image.open(random.choice(self.available_bg)).convert('RGB') / 255),
                          dsize=resize, interpolation=cv2.INTER_NEAREST)

    def __call__(self, src):
        if np.random.uniform(low=0.0, high=1.0) > self.probability:
            return src 
        h, w = src.shape[:2]
        target = self.random_background((h, w))
        resize_dim = int(np.random.uniform(low=self.min_resize_ratio, high=1.0) * h)
        src = cv2.resize(src, dsize=(resize_dim, resize_dim), interpolation=cv2.INTER_NEAREST)
        h, w = src.shape[:2]
        x1 = random.randint(0, target.shape[0] - h)
        y1 = random.randint(0, target.shape[1] - w)
        mask = 0 ** np.ceil(src)
        target[x1:x1 + h, y1:y1 + w, :] = mask * target[x1:x1 + h, y1:y1 + w, :] + src
        return target


class FractionalMaxPool(object):
    def __init__(self, args):
        self.probability = args.fmp_prob
        kernel_size = args.kernel_sizes
        if sum(kernel_size) != 0:
            assert (len(kernel_size) % 2 == 0) and (sum(kernel_size[1::2]) == 1) and (0 not in kernel_size)
            self.kernel_size = lambda: np.random.choice(a=kernel_size[0::2], p=kernel_size[1::2])
        else:
            self.kernel_size = lambda: 3

    def _maxpool(self, src, kernel_size):

        rows, cols = src.shape[:2]
        y = rows // kernel_size
        x = cols // kernel_size
        img_pad = src[:y * kernel_size, : x * kernel_size, ...]
        re_dim = (y, kernel_size, x, kernel_size) + src.shape[2:]
        src = np.nanmax(img_pad.reshape(re_dim),
                        axis=(1, 3)) / 255.

        return src

    def __call__(self, img):
        img = np.array(img * 255, dtype='uint8')
        r = np.arange(np.min(img.shape[:2]))
        n_iters = max(1, np.random.poisson(5))
        for _ in range(n_iters):
            if not np.random.binomial(n=1, p=self.probability):
                break
            idxs1 = sorted(np.random.choice(a=r, replace=True, size=2))
            idxs2 = sorted(np.random.choice(a=r, replace=True, size=2))
            s1 = idxs1[1] - idxs1[0]
            s2 = idxs2[1] - idxs2[0]
            if s1 >= 4 and s2 >= 4:
                img[idxs1[0]:idxs1[1], idxs2[0]:idxs2[1], :] = \
                    np.array(cv2.resize(
                        self._maxpool(img[idxs1[0]:idxs1[1], idxs2[0]:idxs2[1], :],
                                      int(self.kernel_size())), dsize=(s2, s1)) * 255, dtype='uint8')
        return Image.fromarray(img)

