import os
import numpy as np
from glob import glob
from argparse import Namespace
from PIL import Image, ImageFont, ImageDraw, ImageOps


class COLOR_OPTS:
    def __init__(self):
        self.options = [
            'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'gray', 'white'
        ]

    def __call__(self, idx):
        return self.options[idx]


class CHAR_OPTS:
    # NOTE THAT ALL 'W's AND ALL '9's HAVE BEEN REMOVED
    def __init__(self):
        self.options = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '0'
        ]

    def __call__(self, idx):
        return self.options[idx]


class SHAPE_OPTS:
    def __init__(self):
        self.options = [
            'circle', 'semicircle', 'quarter_circle', 'triangle', 'square', 'rectangle',
            'trapezoid', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'star', 'cross'
        ]

    def __call__(self, idx):
        return self.options[idx]


class FONT_OPTS:
    def __init__(self):
        self.options = glob('../data/fonts/*/*.ttf')

    def __call__(self, idx):
        return self.options[idx]


class ATTR:
    def __init__(self):
        self.color_call = COLOR_OPTS()
        self.char_call = CHAR_OPTS()
        self.shape_call = SHAPE_OPTS()
        self.font_call = FONT_OPTS()

    def random_features(self, n, return_index=False):
        shp_clrs, shp_clr_stk, let_clrs, let_clr_stk = self.random_colors(n)
        lets, lets_stk = self.random_letters(n)
        shapes, shp_stk = self.random_shapes(n)
        fnts, fnt_stk = self.random_fonts(n)
        rotations = self.random_rotate(n)
        if return_index:
            return Namespace(**{
                'shape_color': shp_clrs, 'letter_color': let_clrs, 'letter': lets,
                'shape': shapes, 'font': fnts, 'shape_color_idx': shp_clr_stk, 'letter_color_idx': let_clr_stk,
                'letter_idx': lets_stk, 'shape_idx': shp_stk, 'font_idx': fnt_stk,
                'rotation': rotations
            })
        return Namespace(**{
                'shape_color': shp_clrs, 'letter_color': let_clrs, 'letter': lets,
                'shape': shapes, 'font': fnts, 'rotation': rotations
        })

    def random_colors(self, n):
        shape_color = np.random.randint(0, 9, size=(n, ))
        letter_color = np.random.randint(0, 9, size=(n, ))
        col_stack = np.column_stack((shape_color, letter_color))
        for idx, (s, l) in enumerate(col_stack):
            if s == l:
                i0 = np.random.randint(0, 9)
                col_stack[idx, 0] = i0
                col_stack[idx, 1] = list(set(range(9)) - {i0})[np.random.randint(0, 8)]
        colors = np.vectorize(lambda x: self.color_call(x))(col_stack)
        return colors[:, 0], col_stack[:, 0], colors[:, 1], col_stack[:, 1]

    def random_letters(self, n):
        let_stack = np.random.randint(0, 34, size=(n,))
        chars = np.vectorize(lambda x: self.char_call(x))(let_stack)
        return chars, let_stack

    def random_shapes(self, n):
        shp_stack = np.random.randint(0, 13, size=(n,))
        shapes = np.vectorize(lambda x: self.shape_call(x))(shp_stack)
        return shapes, shp_stack

    def random_fonts(self, n):
        fnt_stack = np.random.randint(0, 81, size=(n,))
        fonts = np.vectorize(lambda x: self.font_call(x))(fnt_stack)
        return fonts, fnt_stack

    def random_rotate(self, n):
        return np.random.randint(low=0, high=355, size=(n, ))

    @staticmethod
    def ns_iter(ns, idx):
        return np.array(list(vars(ns).values()))[:, idx]


class DRAWSHAPES:

    @staticmethod
    def circle(draw, res, color):
        square_height = res * np.random.randint(60, 95) / 100
        square_border = (res - square_height) / 2
        top = (square_border, square_border)
        bottom = (res - square_border, res - square_border)
        draw.pieslice([top, bottom], 0, 360, fill=color)

    @staticmethod
    def semicircle(draw, res, color):
        square_height = res * np.random.randint(85, 100) / 100
        square_border = (res - square_height) / 2
        offset = square_height / 4
        top = (square_border, square_border + offset)
        bottom = (res - square_border, res - square_border + offset)
        draw.pieslice([top, bottom], 180, 360, fill=color)

    @staticmethod
    def quarter_circle(draw, res, color):
        square_height = res * np.random.randint(140, 170) / 100
        square_border = (res - square_height) / 2
        offset = square_height / 4
        top = (square_border + offset, square_border + offset)
        bottom = (res - square_border + offset,
                  res - square_border + offset)
        draw.pieslice([top, bottom], 180, 270, fill=color)
        draw.point((0, 0), fill=color)

    @staticmethod
    def square(draw, res, color):
        square_height = res * np.random.randint(55, 90) / 100
        square_border = (res - square_height) / 2
        top = (square_border, square_border)
        bottom = (res - square_border, res - square_border)
        draw.rectangle([top, bottom], fill=color)

    @staticmethod
    def rectangle(draw, res, color):
        if np.random.randint(0, 1) == 0:
            rectangle_width = res * np.random.randint(40, 50) / 100
            rectangle_height = res * np.random.randint(80, 97) / 100
        else:
            rectangle_width = res * np.random.randint(80, 97) / 100
            rectangle_height = res * np.random.randint(40, 50) / 100

        border_width = (res - rectangle_width) / 2
        border_height = (res - rectangle_height) / 2
        top = (border_width, border_height)
        bottom = (res - border_width, res - border_height)
        draw.rectangle([top, bottom], fill=color)

    @staticmethod
    def trapezoid(draw, res, color):
        top_width = res * np.random.randint(40, 50) / 100
        bottom_width = res * np.random.randint(75, 95) / 100
        height = res * np.random.randint(42, 60) / 100

        border_top_width = (res - top_width) / 2
        border_bottom_width = (res - bottom_width) / 2
        border_height = (res - height) / 2

        top_left = (border_top_width, border_height)
        top_right = (res - border_top_width, border_height)

        bottom_left = (border_bottom_width,
                       res - border_height)

        bottom_right = (res - border_bottom_width,
                        res - border_height)
        draw.polygon((top_left, top_right, bottom_right, bottom_left), fill=color)

    @staticmethod
    def star(draw, res, color):
        sides = 5
        cord = res * 55 / 100
        angle = 2 * np.pi / sides
        rotation = np.pi / 2
        points = []
        for s in (0, 2, 4, 1, 3):
            points.append(np.cos(angle * s - rotation) * cord + res / 2)
            points.append(np.sin(angle * s - rotation) * cord + res / 2)
        draw.polygon(points, fill=color)
        rotation = np.pi * 3 / 2
        cord = res * 23 / 100
        points = []
        for s in range(sides):
            points.append(np.cos(angle * s - rotation) * cord + res / 2)
            points.append(np.sin(angle * s - rotation) * cord + res / 2)
        draw.polygon(points, fill=color)

    @staticmethod
    def polygon(draw, res, sides, color):
        cord = res * np.random.randint(45, 50) / 100
        angle = 2 * np.pi / sides
        rotation = 0
        points = []
        if sides % 2 == 1:
            rotation = np.pi / 2
        for s in range(sides):
            points.append(np.cos(angle * s - rotation) * cord + res / 2)
            points.append(np.sin(angle * s - rotation) * cord + res / 2)
        draw.polygon(points, fill=color)

    @staticmethod
    def cross(draw, res, color):
        rectangle_width = res * np.random.randint(35, 40) / 100
        rectangle_height = res * np.random.randint(85, 99) / 100

        border_width = (res - rectangle_width) / 2
        border_height = (res - rectangle_height) / 2
        top = (border_width, border_height)
        bottom = (res - border_width, res - border_height)
        draw.rectangle([top, bottom], fill=color)
        top = (border_height, border_width)
        bottom = (res - border_height, res - border_width)
        draw.rectangle([top, bottom], fill=color)


def HSLConversion(color):

    options = {
        'red': (np.random.randint(0, 4), np.random.randint(50, 100), np.random.randint(40, 60)),
        'orange': (np.random.randint(9, 33), np.random.randint(50, 100), np.random.randint(40, 60)),
        'yellow': (np.random.randint(43, 55), np.random.randint(50, 100), np.random.randint(40, 60)),
        'green': (np.random.randint(75, 120), np.random.randint(50, 100), np.random.randint(40, 60)),
        'blue': (np.random.randint(200, 233), np.random.randint(50, 100), np.random.randint(40, 60)),
        'purple': (np.random.randint(266, 291), np.random.randint(50, 100), np.random.randint(40, 60)),
        'brown': (np.random.randint(13, 20), np.random.randint(25, 50), np.random.randint(22, 40)),
        'black': (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(0, 13)),
        'gray': (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(25, 60)),
        'white': (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(80, 100))
    }
    h, s, l = options[color]
    color_code = 'hsl(%d, %d%%, %d%%)' % (h, s, l)
    return color_code


def draw_shape(shape, draw, res, color):

    options = {
        'circle': lambda d, r, c: DRAWSHAPES.circle(d, r, c),
        'semicircle': lambda d, r, c: DRAWSHAPES.semicircle(d, r, c),
        'quarter_circle': lambda d, r, c: DRAWSHAPES.quarter_circle(d, r, c),
        'triangle': lambda d, r, c: DRAWSHAPES.polygon(d, r, 3, c),
        'square': lambda d, r, c: DRAWSHAPES.square(d, r, c),
        'rectangle': lambda d, r, c: DRAWSHAPES.rectangle(d, r, c),
        'trapezoid': lambda d, r, c: DRAWSHAPES.trapezoid(d, r, c),
        'pentagon': lambda d, r, c: DRAWSHAPES.polygon(d, r, 5, c),
        'hexagon': lambda d, r, c: DRAWSHAPES.polygon(d, r, 6, c),
        'heptagon': lambda d, r, c: DRAWSHAPES.polygon(d, r, 7, c),
        'octagon': lambda d, r, c: DRAWSHAPES.polygon(d, r, 8, c),
        'star': lambda d, r, c: DRAWSHAPES.star(d, r, c),
        'cross': lambda d, r, c: DRAWSHAPES.cross(d, r, c)
    }
    options[shape](draw, res, color)


def draw_text(sample, draw, res, colorcode):
    font = ImageFont.truetype(sample['font'], size=int(res/2))
    w, h = draw.textsize(sample['letter'], font=font)

    pos = ((res - w) / 2, (res - h) / 2)
    draw.text(pos, sample['letter'], fill=colorcode, font=font)


def make_targets(args):
    def draw(sample, res):
        img = Image.new('RGBA', size=args.size[:-1], color=(0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        colorcode_shp = HSLConversion(color=sample['shape_color'])
        draw_shape(shape=sample['shape'], draw=dr, res=res, color=colorcode_shp)
        colorcode_let = HSLConversion(sample['letter_color'])
        draw_text(sample=sample, draw=dr, res=res, colorcode=colorcode_let)
        img = img.rotate(sample['rotation'])
        img = ImageOps.expand(img, border=int(res * 0.5), fill=(0))
        crop_val = int((img.width - args.size[0]) / 2)
        crop_box = (crop_val, crop_val, img.width - crop_val, img.height - crop_val)
        img = img.crop(crop_box)
        img.save(sample['path'])
    def check_dirs():
        if os.path.exists(args.csv_path):
            raise Exception('the CSV path {} already exits.'.format(args.csv_path))
        if os.path.exists(args.img_dir):
            raise Exception('the image directory {} already exits.'.format(args.img_dir))
        else:
            os.mkdir(args.img_dir)
        args.csv_path = './data/CSV/' + args.csv_path
    args.img_dir = './data/generated_targets/' + args.img_dir + '/train'
    if not os.path.exists(args.img_dir):
        os.mkdir(args.img_dir)
    path_template = os.getcwd()[:-8] + args.img_dir[3:] + '/{}.png'
    assert args.n > 0 and len(args.size) == 3 and args.size[-1] == 3
    check_dirs()
    sampler = ATTR()
    samples = pd.DataFrame(vars(sampler.random_features(n=args.n, return_index=True)))
    samples['path'] = np.vectorize(lambda i: path_template.format(i))(np.arange(args.n))
    samples.to_csv(args.csv_path)
    for i, row in samples.iterrows():
        res = np.random.randint(args.lb_size, args.ub_size)
        draw(row, res)
        if i % 100 == 0:
            print('iteration: {}'.format(i))

