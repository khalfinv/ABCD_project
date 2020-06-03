#!/usr/bin/python3
"""
==========================================================================================
Split ABCD subjects (after post-processing ) to 3 datasets with the following percentage:
60% train, 20% validation and 20% test. The ABCD data is augmented to create more volume of data.
==========================================================================================

@Input:  
min_scan_len = integer. Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)
subjects_data_dict = string. path to pkl file containing the subjects data dictionary after post processing
score_file = string. path to excel file containing the behavioural scores
score_key = string. name of the test score column
classification = bool. If flag exists, creates 3 classes instade of regression values
out_folder = string. path to output folder. Default is current folder. 

@Output:
train_set.pkl file: train dataset
validate_set.pkl file: validate dataset 
test_set.pkl file: test dataset
Each dataset is of type List of Tuples. Tuple = (time_series, score) 

"""

import os, sys, pickle, argparse, re
import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def getClassFor2a(score):
	mean = 100
	sd = 15
	below_avg = 1
	normal = 0
	if (score < (mean - sd)):
		score_class = below_avg
	else:
		score_class = normal
	return score_class
def getClassFor2b(score):
	mean = 100
	sd = 15
	below_avg = 0
	above_avg = 1
	if (score < (mean - sd)):
		score_class = below_avg
	elif (score > (mean + sd)):
		score_class = above_avg
	else:
		score_class = -1
	return score_class	
def getClassFor3(score):
	mean = 100
	sd = 15
	below_avg = 0
	avg = 1
	above_avg = 2
	if(score > (mean + sd)):
		score_class = above_avg
	elif (score < (mean - sd)):
		score_class = below_avg
	else:
		score_class = avg
	return score_class
def getClassByHist(score):
	if score < 88:
		score_class = 0
	elif score >= 88 and score < 97:
		score_class = 1
	elif score >= 97 and score < 107:
		score_class = 2
	else:
		score_class = 3
	return score_class
	
def createDataset(data, label_type):
	if label_type == '2a':
		data['label'] = data['label'].apply(getClassFor2a)
	elif label_type == '2b':
		data['label'] = data['label'].apply(getClassFor2b)
		map = data['label'] != -1
		data = data[map]
	elif label_type == '3':
		data['label'] = data['label'].apply(getClassFor3)
	elif label_type == '4':
		data['label'] = data['label'].apply(getClassByHist)
	return data

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', required=True, type=str, help='path to csv file containing the data')
	parser.add_argument('--label_type', required=True, type=str, help='4, 3 or 2a (below vs normal), 2b(above vs below) classes. 0 for regression')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()

	if(args.label_type == "4"):
		print("Created train and test datasets with 4 categorical classes, 0-4. each class frequency is approximately 25%")
	elif(args.label_type == "3"):
		print("Created train and test datasets with 3 categorical classes, 0-3. 0 - children below average, 1 - average, 2 - above average")
	elif(args.label_type == "2a"):
		print("Created train and test datasets with 2 categorical classes, 0-1. 1 - children below average, 0 - children normal range")
	elif(args.label_type == "2b"):
		print("Created train and test datasets with 2 categorical classes, 0-1. 0 - children below average, 1 - children above average")
	elif(args.label_type == "0"):
		print("Created train and test datasets with regression scores")
	else:
		print("Invalid label")
		sys.exit()
		
	data_df = pd.read_csv(args.data)
	data_df = data_df.drop('SUBJECTKEY', axis = 1)
	
	data = createDataset(data_df,args.label_type)
	X = data.drop('label', axis = 1)
	y = data['label']
	X_train, X_test, y_train, y_test = train_test_split(X.values.tolist(), y.values.tolist(), test_size=0.1)
	print("num_of_train: ", len(X_train)," num_of_test: ", len(X_test))
	#Save the datasets
	data.to_csv(path_or_buf=args.out_folder + "/labeled_data.csv", index=False)
	
	train_file = open(args.out_folder + "/train_set.pkl", mode="wb")
	pickle.dump({"X": X_train, "y": y_train}, train_file)
	train_file.close() 

	test_file = open(args.out_folder + "/test_set.pkl", mode="wb")
	pickle.dump({"X": X_test, "y": y_test}, test_file)
	test_file.close() 
 