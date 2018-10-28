import glob
import cv2 
import pickle 
import numpy as np
import torch 
from model import Net
def unpickle(file):
	with open(file, 'rb') as fo: 
		dict = pickle.load(fo)
	return dict 

def load_databatch(data_path, idx):
	data_file = os.path.join(data_path, 'train_data_batch_')
	d = unpickle(data_file + str(idx))
	x = d['data']
	y = d['labels'] 
#	mean_image = d['mean']

	x = x/np.float32(255)
#	mean_image = mean_image
	
	img_size2 = img_size * img_size

	x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
	x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2) # This might be tensorflow language

def preprocess_image(pic): 
	#TODO Preprocessing of image
	return 

def postprocess_image(pic): 
	#TODO Postprocessing i.e. multiplying by 255 
	return


