#!/usr/bin/python3
"""
==========================================================================================
Split ABCD subjects to 3 datasets (time-series (matrix: 264*375), behavioural score (float)
 or 3 classes (according to the user's need)).
train, validation and test
==========================================================================================

@Input:  
dicts_pkl = string. path to pkl file with test-validate-train dictionaries (the post processing dictionary splitted to 3)
score_file = string. path to excel file containing the behavioural scores
score_key = string. name of the test score column
classification = bool. If flag exists, creates 3 classes instade of regression values
out_folder = string. path to output folder. Default is current folder. 

@Output:
train_set.pkl file: train dataset
validate_set.pkl file: validate dataset 
test_set.pkl file: test dataset 	

"""


import os, sys, pickle, argparse, re
import pandas as pd

def createRegDataset(dict, score_file, score_key):
	"""Create dataset - time-series (matrix: 264*375), behavioural score (float)
	param dict: dictionary. Subject's data dictionary from post processing step
	param scores_file: DataFrame . File with subject's behavioural scores and data
	param score_key : string. The wanted score's column name in scores_file
	return: (time_series, scores) : tuple. 
		time_series - list of time-series matrices of size (264, 375), scores - list of behavioural scores
    """
	time_series = []
	scores = []
	for key,value in dict.items():	
		time_series.append((value["time_series"][:375]).T) 	# Get the first 375 frames
		raw = score_file.loc[lambda df: score_file['SUBJECTKEY'] == key] #Get the raw from the excel that match the subject_key. The raw is from type pandas.series
		score = raw[score_key].values[0]
		scores.append(score)
	return (time_series, scores)
	
def createClassDataset(dict, score_file, score_key):
	"""Create dataset - time-series (matrix: 264*375), 3 classes: 0 - below average, 1 - average, 2 - above average
	param dict: dictionary. Subject's data dictionary from post processing step
	param scores_file: DataFrame . File with subject's behavioural scores and data
	param score_key : string. The wanted score's column name in scores_file
	return: (time_series, scores) : tuple. 
		time_series - list of time-series matrices of size (264, 375),
					scores - list of classes : 0 - below average score, 1 - average score, 2 - above average score
    """
	time_series = []
	scores = []
	mean = 100
	sd = 15
	below_avg = 0
	avg = 1
	above_avg = 2
	for key,value in dict.items():	
		time_series.append((value["time_series"][:375]).T) 	# Get the first 375 frames
		raw = score_file.loc[lambda df: score_file['SUBJECTKEY'] == key] #Get the raw from the excel that match the subject_key. The raw is from type pandas.series
		score = raw[score_key].values[0]
		if(score > (mean + sd)):
			scores.append(above_avg)
		elif (score < (mean - sd)):
			scores.append(below_avg)
		else:
			scores.append(avg)
	return (time_series, scores)
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dicts_pkl', required=True, type=str, help='path to pkl file with test-validate-train dictionaries')
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the scores')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	parser.add_argument('--classification', help='creates 3 classes instage of regression values',action='store_true')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()

	#open and read the pkl file of train-validate-test dictionaries
	pkl_file = open(args.dicts_pkl, 'rb')
	all_dicts = pickle.load(pkl_file)
	pkl_file.close()

	train_dict = all_dicts['train_dict']
	validate_dict = all_dicts['validate_dict']
	test_dict = all_dicts['test_dict']

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

	#create datasets
	if args.classification == True:
		train_set = createClassDataset(train_dict, df, args.score_key)
		train_file_name = "train_set_class.pkl"
		validate_set = createClassDataset(validate_dict, df, args.score_key)
		validate_file_name = "validate_set_class.pkl"
		test_set = createClassDataset(test_dict, df, args.score_key)
		test_file_name = "test_set_class.pkl"
		
	else:
		train_set = createRegDataset(train_dict, df, args.score_key)
		train_file_name = "train_set_reg.pkl"
		validate_set = createRegDataset(validate_dict, df, args.score_key)
		validate_file_name = "validate_set_reg.pkl"
		test_set = createRegDataset(test_dict, df, args.score_key)
		test_file_name = "test_set_reg.pkl"
	
	#Save the datasets	
	train_file = open(args.out_folder + '/' + train_file_name, mode="wb")
	pickle.dump(train_set, train_file)
	train_file.close() 

	validate_file = open(args.out_folder + '/' + validate_file_name, mode="wb")
	pickle.dump(validate_set, validate_file)
	validate_file.close() 

	test_file = open(args.out_folder + '/' + test_file_name, mode="wb")
	pickle.dump(test_set, test_file)
	test_file.close() 
