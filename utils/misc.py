import os
import colored
from colored import stylize


idxmap = {
        'shape': {
            'circle': 0, 'semicircle': 1, 'quarter_circle': 2, 'triangle': 3, 'square': 4, 'rectangle': 5,
            'trapezoid': 6, 'pentagon': 7, 'hexagon': 8, 'heptagon': 9, 'octagon': 10, 'star': 11, 'cross': 12
        },
        'color': {'red': 0, 'orange': 1, 'yellow': 2, 'green': 3, 'blue': 4,
                  'purple': 5, 'brown': 6, 'gray': 7, 'white': 8},
        'letter': {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
            'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
            'X': 22, 'Y': 23, 'Z': 24, '1': 25, '2': 26, '3': 27, '4': 28, '5': 29, '6': 30, '7': 31, '8': 32, '0': 33
        }}


def name_model(args):
    model_name = '{subdir}/{args.dataset}/{args.model}.e-{args.n_epochs}.bs-{args.batch_size}'.format(subdir='../tests', args=args)
    if args.message:
        model_name += '.{}'.format(args.message)
    while os.path.exists(model_name):
        model_name += 'i'
    if args.save:
        required_dirs = [
            '{subdir}'.format(subdir='../tests'), '{subdir}/{dataset}'.format(subdir='../tests', dataset=args.dataset),
            '{model_name}'.format(model_name=model_name)]
        for dir in required_dirs:
            if not os.path.exists(dir):
                print(stylize('Making directory: {dir}'.format(dir=dir), colored.fg('red')))
                os.mkdir(dir)
    return model_name
