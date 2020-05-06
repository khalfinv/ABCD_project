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

	
def getScore(score):
	mean = 100
	sd = 10
	below_avg = 0
	above_avg = 1
	avg = 2
	if(score > (mean + sd)):
		score = above_avg
	elif (score < (mean - sd)):
		score = below_avg
	else:
		score = avg
	return score
def createDataset(score_file, score_X, score_Y):
	scores_X = []
	scores_Y = []
	num_of_subjects=0
	new_score_file = score_file[score_X+[score_Y]].dropna()
	for _,raw in new_score_file.iterrows():
		score_class = getScore(raw[score_Y])
		# if(score_class == 2):
			# continue
		scores_Y.append(score_class)
		scores_X.append(raw[score_X].to_list())

	return scores_X, scores_Y

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the scores')
	parser.add_argument('--score_X', required=True,  nargs='+', type=str, help='name of the test score column for X data')
	parser.add_argument('--score_Y', required=True, type=str, help='name of the test score column for Y data')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()


	#Read the excel file 
	df = pd.read_excel(io=args.score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in df):
		print("'SUBJECTKEY' column does not exists in excel file")
		sys.exit()

	if(args.score_Y not in df):
		print(args.score_key, " column does not exist in excel file")
		sys.exit()
	for test in args.score_X:
		if(test not in df):
			print(test, " column does not exist in excel file")
			sys.exit()

	X,y = createDataset(df, args.score_X, args.score_Y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
	print("num_of_train: ", len(X_train)," num_of_test: ", len(X_test))
		
	#Save the datasets	
	train_file = open(args.out_folder + "/train_set.pkl", mode="wb")
	pickle.dump({"X": X_train, "y": y_train}, train_file)
	train_file.close() 

	test_file = open(args.out_folder + "/test_set.pkl", mode="wb")
	pickle.dump({"X": X_test, "y": y_test}, test_file)
	test_file.close() 
 