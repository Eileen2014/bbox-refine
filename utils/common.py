import os
import json
import numpy as np

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
