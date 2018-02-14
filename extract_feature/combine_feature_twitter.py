import sys, os, json, time, datetime
sys.path.append('..')
import utils

project_folder = os.path.join('..', '..')

import numpy as np

def main():
	x15 = np.load(os.path.join(project_folder, 'feature', 'twitter15', 'x.npy'))
	y15 = np.load(os.path.join(project_folder, 'feature', 'twitter15', 'y.npy'))
	x16 = np.load(os.path.join(project_folder, 'feature', 'twitter16', 'x.npy'))
	y16 = np.load(os.path.join(project_folder, 'feature', 'twitter16', 'y.npy'))
	print(x15.shape, x16.shape)
	x = np.concatenate((x15, x16), axis = 0)
	y = np.concatenate((y15, y16), axis = 0)
	print(x.shape, y.shape)
	np.save(os.path.join(project_folder, 'feature', 'twitter', 'x.npy'), x)
	np.save(os.path.join(project_folder, 'feature', 'twitter', 'y.npy'), y)

	
if __name__ == '__main__':
	main()
