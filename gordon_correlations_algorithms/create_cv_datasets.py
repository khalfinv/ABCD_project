#!/usr/bin/python3
"""
==========================================================================================
Split data to K folds and K combinations of folds of K-1 training and 1 validation
==========================================================================================

@Input:  
K = integer. Number of folds
dataset = string. path to pkl file containing all training data
out_folder = string. path to output folder. Default is current folder. 

@Output:
K folders. In each folder there will two pkl files:
	train.pkl and validate.pkl containing the data for training and testing

"""

import os, sys, pickle, argparse, re
import random
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--K', required=False, type=int, default = 10, help='number of folds')
	parser.add_argument('--dataset', required=True, type=str, help='path to pkl file containing all training data')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()

	#load dataset
	pkl_file = open(args.dataset, 'rb')
	subjects = pickle.load(pkl_file)
	X = subjects["X"]
	y = subjects["y"]	
	pkl_file.close()

	
	kf = KFold(n_splits=args.K) # Define the split
	i = 1
	for train_index, test_index in kf.split(X):
		X_train = [X[i] for i in train_index]
		y_train = [y[i] for i in train_index]
		X_test = [X[i] for i in test_index]
		y_test = [y[i] for i in test_index]
		dir_path = args.out_folder + "/cv"+str(i)
		if (os.path.exists(dir_path) == False):
			os.mkdir(dir_path)
		i+=1
		
		# transform the train dataset
		print(Counter(y_train)[0])
		dict_over = {0:Counter(y_train)[0], 1:Counter(y_train)[1], 2:int(Counter(y_train)[1]*0.5), 3:int(Counter(y_train)[1]*0.5)}
		over = SMOTE(sampling_strategy=dict_over)
		X_train, y_train = over.fit_resample(X_train, y_train)
		dict_under = {0:Counter(y_train)[0], 1:int(Counter(y_train)[1]*0.7), 2:Counter(y_train)[2], 3:Counter(y_train)[3]}
		under = RandomUnderSampler(sampling_strategy=dict_under)
		X_train, y_train = under.fit_resample(X_train, y_train)
		
		# summarize the new class distribution
		counter_train = Counter(y_train)
		print("Train: ", counter_train)
		counter_test = Counter(y_test)
		print("Test: ", counter_test)
		#Save the datasets	
		train_file = open(dir_path + "/train_set.pkl", mode="wb")
		pickle.dump({"X": X_train, "y": y_train}, train_file)
		train_file.close() 

		validate_file = open(dir_path+ "/validate_set.pkl", mode="wb")
		pickle.dump({"X": X_test, "y": y_test}, validate_file)
		validate_file.close() 
 
		
		
 