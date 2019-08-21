#!/usr/bin/python3
"""
==========================================================================================
Split ABCD subjects (after post-processing ) to 3 dictionaries with the following percentage:
60% train, 20% validation and 20% test
==========================================================================================

@Input:  
exclusion_criteria = integer. Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)
subjects_data_dict = string. path to pkl file containing the subjects data dictionary after post processing
out_folder = string. path to output folder. Default is current folder. 

@Output:
split_dicts.pkl file: a dictionary with 3 dictionaries with the following format
					all_dicts = {
						'train_dict': train_dict,
						'validate_dict': validate_dict,
						'test_dict': test_dict
					}

"""

import os, sys, pickle, argparse
import math

def splitToDicts(subjects_dict):
	"""Split ABCD subjects (after post-processing ) to 3 dictionaries with the following percentage:
		60% train, 20% validation and 20% test
	param subjects_dict: dictionary. Subject's data dictionary from post processing step
	return: train_dict: dictionary. Subjects for training
			validate_dict: dictionary. Subjects for validation
			test_dict: dictionary. Subjects for testing
    """
	num_of_train = math.floor(len(subjects_dict)*0.6)
	num_of_validate = math.floor(len(subjects_dict)*0.2)
	num_of_test = len(subjects_dict) - (num_of_train + num_of_validate)

	train_dict = {k:subjects_dict[k] for k in list(subjects_dict)[:num_of_train]}
	validate_dict = {k:subjects_dict[k] for k in list(subjects_dict)[num_of_train : (num_of_train + num_of_validate)]}
	test_dict = {k:subjects_dict[k] for k in list(subjects_dict)[-num_of_test:]}
	print("num_of_train: ", num_of_train, " num_of_validate: " , num_of_validate, " num_of_test: ", num_of_test)
	return train_dict, validate_dict, test_dict  
	
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--exclusion_criteria', required=False, type=int, default=375, help='Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)')
	parser.add_argument('--subjects_data_dict', required=True, type=str, help='path to pkl file containing the subjects data dictionary')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()

	#open and read the pkl file of all subjects' data	
	pkl_file = open(args.subjects_data_dict, 'rb')
	subjects_data_dict = pickle.load(pkl_file)
	pkl_file.close()
	
	
	#Create dictionary with only included subject - num_of_volumes >= exclusion_criteria
	only_included_subjects = {}
	for key, val in subjects_data_dict.items():
		if val["num_of_volumes"] >= args.exclusion_criteria:
			only_included_subjects[key] = val
	
	#split the subjects to 3 datasets
	train_dict, validate_dict, test_dict = splitToDicts(only_included_subjects)
		
	all_dicts = {
	'train_dict': train_dict,
	'validate_dict': validate_dict,
	'test_dict': test_dict
	}

	#save to pkl
	f = open(args.out_folder + '/split_dicts.pkl', mode="wb")
	pickle.dump(all_dicts, f)
	f.close() 
 