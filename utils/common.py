import os
import json
import numpy as np
from scipy import ndimage

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def mkdirs(paths):
	if isinstance(paths, list):
		for path in paths:
			mkdir(path)

def read_file(path):
	if not os.path.exists(path):
		raise FileNotFoundError('File: {} doesn\'t exist'.format(path))
	with open(path, 'r') as f:
		lines = f.readlines()
	lines = [path.strip() for path in lines]
	return np.array(lines)

def log_config(flags, path):
	""" Log the model configurations 
	
	Args:
		flags: dictionary containing flags and values
        path : location where the flags are to be saved
	"""
	with open(path, 'w') as f:
		json.dump(flags, f, indent=5)

def get_tightest_bbox(mask):
    """ Generates the tightest bounding box that encloses the 
        given mask
    """
    try:
        slice_y, slice_x = ndimage.find_objects(mask > 0)[0]
    except IndexError:
        print('No mask at all? Weird')
        return -1
    y_min, y_max = slice_y.start, slice_y.stop
    x_min, x_max = slice_x.start, slice_x.stop

    return np.array([y_min, x_min, y_max, x_max])

def get_tightest_bboxes(masks):
    boxes = np.zeros((len(masks), 4))
    for idx, mask in enumerate(masks):
        bbox = get_tightest_bbox(mask)
        if isinstance(bbox, int):
            # FIXME: Shouldn't be happening? Check back later on
            continue
        else:
            boxes[idx] = get_tightest_bbox(mask)
    return boxes

