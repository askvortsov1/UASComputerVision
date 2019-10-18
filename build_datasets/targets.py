import argparse
from utils import image_generation


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', default='targets_csv.csv', type=str,
                    help='all CSVs will be saved to directory ./data/CSV')
parser.add_argument('--img_dir', default='60000', type=str,
                    help='all image directories will be saved to directory ./data/generated_targets')
parser.add_argument('--n', default=50000, type=int, help='how many instances to generate')
parser.add_argument('--size', default=[180, 180, 3], nargs='+', type=int)
parser.add_argument('--lb_size', default=130, type=int,
                    help='lowerbound limit for size of the shape')
parser.add_argument('--ub_size', default=190, type=int,
                    help='upperbound limit for size of the shape')

if __name__ == '__main__':
    args = parser.parse_args()
    image_generation.make_targets(args)



