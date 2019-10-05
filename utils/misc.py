import os
import torch

import numpy as np
import pandas as pd

import colored
from colored import stylize


def name_model(args):
    version = ''
    if args.model == 'WideResnet':
        version = 'width-{}.depth-{}.'.format(args.width, args.depth)
    elif args.model == 'densenetBC':
        version = 'depth-{}.gr-{}.reduce-{}.bttlnk-{}.'.format(
            args.depth, args.growth_rate, args.reduce, args.bottleneck)
    elif args.model == 'shakeshake':
        version = 'depth-{}.base_channels-{}.SI-{}.SF-{}.SB-{}.'.format(
            args.depth, args.base_channels, args.shake_image, args.shake_forward, args.shake_backward)
    elif args.model == 'shake_pyramidnet':
        version = 'depth-{}.alpha-{}.'.format(args.depth, args.alpha)
    elif args.model == 'pyramidnet':
        version = 'depth-{}.alpha-{}.bttlnk-{}.'.format(args.depth, args.alpha, args.bottleneck)
    model_name = '{subdir}/{model}/{dataset}/{version}epochs-{n_epochs}'.format(
        subdir=args.subdir, model=args.model, dataset=args.dataset, version=version, n_epochs=args.n_epochs)
    if args.message:
        model_name += '.{}'.format(args.message)
    while os.path.exists(model_name):
        model_name += 'i'
    if args.save:
        required_dirs = [
            '{subdir}'.format(subdir=args.subdir), '{subdir}/{model}'.format(subdir=args.subdir, model=args.model),
            '{subdir}/{model}/{dataset}'.format(subdir=args.subdir, model=args.model, dataset=args.dataset),
            '{model_name}'.format(model_name=model_name)]
        for dir in required_dirs:
            if not os.path.exists(dir):
                print(stylize('Making directory: {dir}'.format(dir=dir), colored.fg('red')))
                os.mkdir(dir)
    return model_name


def save_model(args, model_name, best_acc, stats, state, improvement, epoch, final_epoch=False):
    if not args.save:
        return 0
    else:
        if improvement and not args.data_only:
            torch.save(state, '{}/ckpt.pth'.format(model_name))
        if final_epoch:
            if not args.data_only:
                os.system('mv {model_name}/ckpt.pth {model_name}/test_acc{acc}.pth'.format(
                    model_name=model_name, acc=np.round(best_acc, 5)))
            pd.DataFrame({
                'test_acc': stats.test_acc, 'test_loss': stats.test_loss
            }).to_csv('{}/test.csv'.format(model_name))
            pd.DataFrame({'train_acc': stats.train_acc, 'train_loss': stats.train_loss}).to_csv('{}/train.csv'.format(model_name))


def log(path, string):
    with open(path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)