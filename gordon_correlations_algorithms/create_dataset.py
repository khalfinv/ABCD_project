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

def splitToDatasets(all_subjects):
	"""Split ABCD subjects (after post-processing ) to 3 datasets with the following percentage:
		60% train, 20% validation and 20% test
	param all_subjects: List of Tuples. Each tuple = (time_series, score)
	return: train_dataset: List of Tuples. Subjects for training
			validate_dataset: List of Tuples. Subjects fot validation
			test_dataset: List of Tuples. Subjects for testing
    """
	num_of_train = math.floor(len(all_subjects)*0.8)
	num_of_validate = math.floor(len(all_subjects)*0.1)
	num_of_test = len(all_subjects) - (num_of_train + num_of_validate)
	random.shuffle(all_subjects) 

	train_dataset = all_subjects[:num_of_train]
	validate_dataset = all_subjects[num_of_train : (num_of_train + num_of_validate)]
	test_dataset = all_subjects[-num_of_test:]
	print("num_of_train: ", num_of_train, " num_of_validate: " , num_of_validate, " num_of_test: ", num_of_test)
	return train_dataset, validate_dataset, test_dataset  
	
def getScore(subject_id, score_file, score_key, is_class):
	"""Get the score's class or actual value.
        classes : below average = 0, average = 1, above average = 2. Mean = 100, sd = 15.
	param subject_id: String. The subject's key.
    param score_file: Dataframe. The scores excel file
    param score_key: String. The score's column name
    param is_class: Bool. If true = split to classes, else return the actual score value
	return: Integer. Score value. 
    """
	raw = score_file.loc[lambda df: score_file['SUBJECTKEY'] == subject_id] #Get the raw from the excel that match the subject_key. The raw is from type pandas.series
	score = -1
	if(raw.empty == False):
		score = int(raw[score_key].values[0])
		# score_intervals = [range(62,91),range(91,98), range(98,172)]
		# a = 0
		# b = 1
		# c = 2
		# if (score in score_intervals[a]):
			# score = a
		# elif (score in score_intervals[b]):
			# score = b
		# elif (score in score_intervals[c]):
			# score = c
		# else:
			# print("error with score: ", score)
		mean = 100
		sd = 10
		below_avg = 0
		avg = 1
		above_avg = 2
		if is_class == True:
			if(score > (mean + sd)):
				score = above_avg
			elif (score < (mean - sd)):
				score = below_avg
			else:
				score = avg
	return score
def createDataset(corr_file, score_file, score_key, is_class):
	"""Create augmented datasets. Each data with num of volumes >= min_scan_len, is augmented 5 times. 
		The times series are cut to min_scan_len, when the start point of the cutting is randomized 5 times.
	param subjects_dict: dictionary. The ABCD subjects' dictionary after post processing.
	param min_scan_len: integer. Number of volumes for inclusion. 
	param score_file: Dataframe. The scores excel file
	param score_key: String. The score's column name
	param is_class: Bool. If true = split to classes, else return the actual score value
	return: List of Tuples. Tuple = (time_series, score). All data aftrer augmentation. 
	"""

	correlations = []
	scores = []
	num_of_subjects=0
	for index, row in corr_file.iterrows():
		corr = row.tolist()[1:]	
		score = getScore(row['SUBJECTKEY'], score_file, score_key, is_class)
		if(score == 1 or score == -1):
			continue
		correlations.append(corr)
		scores.append(score)
	return list(zip(correlations, scores))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--corr_file', required=True, type=str, help='path to excel file containing Gordon correlations')
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the behavioural scores')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	parser.add_argument('--classification', help='creates 3 classes instade of regression values',action='store_true')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()


	#Read the excel files 
	df_corr = pd.read_excel(io=args.corr_file)
	df_scores = pd.read_excel(io=args.score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in df_corr):
		print("'SUBJECTKEY' column does not exists in the correlations excel file")
		sys.exit()
	if('SUBJECTKEY' not in df_scores):
		print("'SUBJECTKEY' column does not exists in the scores excel file")
		sys.exit()

	if(args.score_key not in df_scores):
		print(args.score_key, " column does not exist in scores excel file")
		sys.exit()
		
	df_corr = df_corr.dropna()
	df_scores = df_scores.dropna(subset=['NIHTBX_FLANKER_AGECORRECTED'])
	subjects = createDataset(df_corr, df_scores, args.score_key, args.classification)

	#split the subjects to 3 datasets
	train_set, validate_set, test_set = splitToDatasets(subjects)

	if args.classification == True:
		suffix = "class"		
	else:
		suffix = "reg"

	#Save the datasets	
	train_file = open(args.out_folder + "/train_set_" + suffix + ".pkl", mode="wb")
	pickle.dump(train_set, train_file)
	train_file.close() 

	validate_file = open(args.out_folder + "/validate_set_" + suffix + ".pkl", mode="wb")
	pickle.dump(validate_set, validate_file)
	validate_file.close() 

	test_file = open(args.out_folder + "/test_set_" + suffix + ".pkl", mode="wb")
	pickle.dump(test_set, test_file)
	test_file.close() 
	
	
	

		

 