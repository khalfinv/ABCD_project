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
import shutil


def createDataset(score_file, score_X, score_Y):
	random.seed()
	scores_X = []
	scores_Y = []
	num_of_subjects=0
	new_score_file = score_file[score_X+[score_Y]].dropna()
	for _,raw in new_score_file.iterrows():
		score_class = getScore(raw[score_Y])
		if(score_class == 1):
			continue
		scores_Y.append(score_class)
		scores_X.append(raw[score_X].to_list())

	return list(zip(scores_X, scores_Y))

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
		if (os.path.exists(dir_path) == True):
			shutil.rmtree(dir_path)
		os.mkdir(dir_path)
		i+=1
		#Save the datasets	
		train_file = open(dir_path + "/train_set.pkl", mode="wb")
		pickle.dump({"X": X_train, "y": y_train}, train_file)
		train_file.close() 

		validate_file = open(dir_path+ "/validate_set.pkl", mode="wb")
		pickle.dump({"X": X_test, "y": y_test}, validate_file)
		validate_file.close() 
 
		
		
 