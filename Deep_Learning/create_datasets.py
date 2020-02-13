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
	num_of_train = math.floor(len(all_subjects)*0.6)
	num_of_validate = math.floor(len(all_subjects)*0.2)
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
	score = raw[score_key].values[0]
	mean = 100
	sd = 15
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
def createAugDataset(subjects_dict, min_scan_len, score_file, score_key, is_class):
	"""Create augmented datasets. Each data with num of volumes >= min_scan_len, is augmented 5 times. 
        The times series are cut to min_scan_len, when the start point of the cutting is randomized 5 times.
    param subjects_dict: dictionary. The ABCD subjects' dictionary after post processing.
    param min_scan_len: integer. Number of volumes for inclusion. 
    param score_file: Dataframe. The scores excel file
    param score_key: String. The score's column name
    param is_class: Bool. If true = split to classes, else return the actual score value
	return: List of Tuples. Tuple = (time_series, score). All data aftrer augmentation. 
    """
	random.seed()
	time_series = []
	scores = []
	for key, val in subjects_data_dict.items():
		score = getScore(key, score_file, score_key, is_class) 
		rand_times = 5
		# print("subject:", key)
		# print("num_of_volumes:", val["num_of_volumes"])
		if(val["num_of_volumes"] >= min_scan_len):
			while (rand_times > 0):
				start = random.randrange(val["num_of_volumes"] - min_scan_len + 1)
				# print("rand:", start)
				time_series.append((val["time_series"][start:(start+min_scan_len)]).T)
				scores.append(score)
				# print("new data inserted")
				rand_times-=1
	return list(zip(time_series, scores))
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--min_scan_len', required=False, type=int, default=375, help='Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)')
	parser.add_argument('--subjects_data_dict', required=True, type=str, help='path to pkl file containing the subjects data dictionary')
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the scores')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	parser.add_argument('--classification', help='creates 3 classes instage of regression values',action='store_true')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()

	#open and read the pkl file of all subjects' data	
	pkl_file = open(args.subjects_data_dict, 'rb')
	subjects_data_dict = pickle.load(pkl_file)
	pkl_file.close()
	
	#Read the excel file 
	df = pd.read_excel(io=args.score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in df):
		print("'SUBJECTKEY' column does not exists in excel file")
		sys.exit()

	if(args.score_key not in df):
		print(args.score_key, " column does not exist in excel file")
		sys.exit()

	#change the SUBJECT_KEY values to match to SUBJECT_KEY format in the subjects_data_dict (without "_")
	for i, row in df.iterrows(): 
		df.at[i,'SUBJECTKEY'] = re.sub(r"[^a-zA-Z0-9]","",df.at[i,'SUBJECTKEY'])

	subjects = createAugDataset(args.subjects_data_dict,args.min_scan_len, df, args.score_key, args.classification)
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
	
	
	

		

 