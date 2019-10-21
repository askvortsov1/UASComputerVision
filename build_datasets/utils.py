import os
import pickle
import numpy as np


def check_collisions(bounding_boxes, new_bounds):
    i1, i2, j1, j2 = new_bounds
    i, j = np.mean([i1, i2]), np.mean([j1, j2])
    dist = abs(i1 - i2)
    for b in bounding_boxes:
        m1, m2 = np.mean([b[0], b[1]]), np.mean([b[2], b[3]])
        if np.sqrt(np.power(i - m1, 2) + np.power(j - m2, 2)) < 2 * dist:
            return False
    return True


def save_boxes(boxes_dict, save_loc, split=None):
    if split is not None:
        pickle_file = '{}/boxes_{}.pkl'.format(save_loc, split)
    else: 
        pickle_file = '{}/boxes.pkl'.format(save_loc)
    with open(pickle_file, 'wb') as handle:
        pickle.dump(boxes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved bounding boxes to {}'.format(pickle_file))