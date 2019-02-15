import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from skimage import feature as ft
import os
import pandas as pd 
from tempfile import TemporaryFile
from sklearn.decomposition import PCA


def image_feature_label(img,dir_name):
	img = cv2.imread(img)
	img = cv2.resize(img, (256, 256))
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	features = ft.hog(img,  # input image
	                  orientations=18,  # number of bins
	                  pixels_per_cell=(16,16),  # pixel per cell
	                  cells_per_block=(16,16),  # cells per block
	                  block_norm='L1',
	                  transform_sqrt=True,  # power law compression (also known as gamma correction)
	                  feature_vector=True,  # flatten the final vectors
	                  visualize=True)  # return HOG map
	return features[0], dir_name, features[1]

def generate_features():
	feature_list = []
	label_list = []
	dirpath = "./boat-types-recognition/"
	dir_list = os.listdir(dirpath)
	for dir_name in dir_list:
		if not dir_name.startswith('.'):
			folder_path = dirpath + dir_name + "/"
			file_list = os.listdir(folder_path)
			for image in file_list:
				if not image.startswith('.'):
					image_path = folder_path + image
					features, label, feature_image = image_feature_label(image_path,dir_name)
					feature_list.append(features)
					label_list.append(label)
	return np.array([feature_list, label_list])

def save_feature_file():
	features = generate_features()
	np.save("type_feature",features)

def load_feature_file():
	feature,label = np.load("type_feature.npy")
	return feature,label


#save_feature_file()
#print ('xxx')
feature,label = load_feature_file()
#print (label)
def PCA_Algorithm(data,d):
	pca = PCA(n_components = 1)
	pca.fit(data)
	#print(pca.explained_variance_ratio_)
	#print(pca.singular_values_)  
PCA_Algorithm(feature,1)

















