import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from skimage import feature as ft
import os
import pandas as pd 
from tempfile import TemporaryFile
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import cv2
def image_feature_label(img,dir_name):
	""" use hog to generate image features
    Args:
	     img : image name/image path
	     dir_name: boat type folder, use to makde labels
	Returns:
        features[0]: feature array 
        dif_name : label name
        feature[1] : hig image 
	"""
	img = cv2.imread(img)
	img = cv2.resize(img, (256, 256))
	features = ft.hog(img,  # input image
	                  orientations=18,  # number of bins
	                  pixels_per_cell=(16,16),  # pixel per cell
	                  cells_per_block=(16,16),  # cells per block
	                  block_norm='L1',
	                  transform_sqrt=True,  # power law compression (also known as gamma correction)
	                  feature_vector=True,  # flatten the final vectors
	                  visualize=True)  # return HOG map
	return features[0], dir_name, features[1]

def generate_features(path):
	""" generate all image features and labels
    Args:
	     path : path of training data folder and testing data folder
	Returns:
        feature and labnel array
	"""
	feature_list = []
	label_list = []
	dirpath = path
	dir_list = os.listdir(dirpath) # read all boat folders under training/testing folder
	for dir_name in dir_list:
		if not dir_name.startswith('.'):
			folder_path = dirpath + dir_name + "/"
			file_list = os.listdir(folder_path) # read all image  under boat type folder
			for image in file_list:
				if not image.startswith('.'):
					image_path = folder_path + image
					features, label, feature_image = image_feature_label(image_path,dir_name)
					feature_list.append(features)
					label_list.append(label)
	return np.array(feature_list),np.array(label_list)

def image_feature_label_wave(img,dir_name):
	""" use hog to generate image features
    Args:
	     img : image name/image path
	     dir_name: boat type folder, use to makde labels
	Returns:
        features[0]: feature array 
        dif_name : label name
        feature[1] : hig image 
	"""
	img = cv2.imread(img)
	img = cv2.resize(img, (256, 256))
	coeffs = pywt.dwt2(img, 'haar')
	coeffs = pywt.idwt2(coeffs, 'haar')

	return coeffs[0], dir_name

def generate_wave(path):
	feature_list = []
	label_list = []
	dirpath = path
	dir_list = os.listdir(dirpath) # read all boat folders under training/testing folder
	for dir_name in dir_list:
		if not dir_name.startswith('.'):
			folder_path = dirpath + dir_name + "/"
			file_list = os.listdir(folder_path) # read all image  under boat type folder
			for image in file_list:
				if not image.startswith('.'):
					image_path = folder_path + image
					features, label = image_feature_label_wave(image_path,dir_name)
					feature_list.append(features)
					label_list.append(label)
	return np.array(feature_list),np.array(label_list)



def save_feature_file():
	""" 
	save generated feature array to csv file
	"""
	training_features,training_label = generate_wave("./training_data/")
	np.save("training_data",training_features)
	np.save("training_label",training_label)


def load_training_file():
	""" 
	read generated feature array from csv file
	"""
	feature = np.load("training_data.npy")
	label = np.load("training_label.npy")
	return feature,label

def load_testing_file():
	feature = np.load("test_data.npy")
	label = np.load("test_label.npy")
	return feature,label


def PCA_Algorithm(data,d):
	'''used pca method to reduce features'''
	pca = PCA(n_components = d)
	pca.fit(data)
	final_data = pca.transform(data)
	return final_data

def PCA_Algorithm_test(training_data, training_label, test_data, test_label,d):
	'''used pca method to reduce features'''
	pca = PCA(n_components = d)
	pca.fit(training_data)
	pca.fit(test_data)
	final_training = pca.transform(training_data)
	final_test = pca.transform(test_data)	
	return final_training, final_test

# best POV is 472
def calculate_POV(data):
    mean = np.mean(data,axis=0)
    normal_data = data - mean
    cov_matrix = np.cov(normal_data.T)  # calculate the cov
    eig_value, eig_vector=  np.linalg.eig(cov_matrix)  
    feature_rank = np.argsort(-eig_value) 
    sum_eig =  sum(eig_value) * 0.95
    count = 0
    sum_count = 0
    for index in feature_rank:
    	if sum_count < sum_eig:
    		sum_count += eig_value[index]
    		count +=1
    return count


def KNN_classifier(training_data, training_label, test_data, test_label, n):
	neigh = KNeighborsClassifier(n_neighbors=n)
	neigh.fit(training_data, training_label) 
	test_result = neigh.predict(test_data)
	test_score = neigh.score(test_data,test_label)
	return test_result,test_score
	
def decision_tree(training_data, training_label, test_data, test_label):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(training_data, training_label)
	test_result = clf.predict(test_data)
	test_score = clf.score(test_data,test_label)
	print (test_score)

def feature_selection(training_data, training_label, d):

	X_new = SelectKBest(chi2, k=d)
	train_data = X_new.fit_transform(training_data, training_label)
	return train_data

def KNN_cross_validation(training_data,training_label):
	k_range = range(10,11)
	k_scores = []
	for k in k_range:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, training_data, training_label, cv=10, scoring='accuracy')
		k_scores.append(scores.mean())
		print (scores.mean())

''' 
for n = 30, the accuracy is 0.4740
for n = 11, the accuracy is 0.5240
for n = 5, the accuracy is 0.49
'''
#result, score = KNN_classifier(training_data,training_label,test_data,test_label, 5)
'''
use pca first, then use knn classiffier
let n = 11
d = 10, score = 0.2406
d = 5 , score = 0.28877
'''
save_feature_file()

training_data, training_label = load_training_file()
feature_range = range(200,300)
for f in feature_range:
	train=  feature_selection(training_data, training_label,f)
	KNN_cross_validation(train,training_label)
print ('done')


#print (result)
#print (score)
#print (calculate_POV(feature))
#final_data = PCA_Algorithm(feature,1)
#print (final_data.shape)
















