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
import pandas as pd
from Gordon_networks import networks_list


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--corr_file', required=True, type=str, help='path to excel file containing Gordon correlations')
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the behavioural scores')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	parser.add_argument('--covar_file', required=False, type=str, help='path to covariate file')
	parser.add_argument('--covar', required=False, type=str, help='name of the covariate column')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()


	#Read the excel files 
	df_corr = pd.read_excel(io=args.corr_file)
	df_scores = pd.read_excel(io=args.score_file)
	#Read the Y excel file
	df_scores = df_scores[['SUBJECTKEY'] + [args.score_key]]
	print(df_scores.shape)
	print(df_corr.shape)
	data_df = pd.merge(left=df_corr, right=df_scores, left_on='SUBJECTKEY', right_on='SUBJECTKEY').dropna()
	if(args.covar_file != None):
		df_covar = pd.read_excel(io=args.covar_file)
		df_covar = df_covar[['SUBJECTKEY'] + [args.covar]]
		data_df = pd.merge(left=data_df, right=df_covar, left_on='SUBJECTKEY', right_on='SUBJECTKEY').dropna()
	data_df = data_df.drop('SUBJECTKEY', axis = 1)
	cognitive_networks = ["CO-CO","CO-DMN","CO-DAN","CO-FP","CO-Salience", "CO-VAN","DMN-DMN","DMN-DAN", "DMN-FP","DMN-Salience", "DMN-VAN",
	"DAN-DAN", "DAN-FP","DAN-Salience", "DAN-VAN", "FP-FP","FP-Salience","FP-VAN","Salience-Salience","Salience-VAN","VAN-VAN"]
	indexes = [] 
	for net in cognitive_networks:
		indexes.append(networks_list.index(net))
	label_index = data_df.columns.get_loc(args.score_key)
	indexes.append(label_index)
	if(args.covar != None):
		covar_index = data_df.columns.get_loc(args.covar)
		indexes.append(covar_index)
		data_df.rename(columns={args.covar:'covar'}, inplace=True)
	
	data_df.rename(columns={args.score_key:'label'}, inplace=True)
	data_df = data_df.iloc[:,indexes]
	
	#Save the data
	data_df.to_csv(path_or_buf=args.out_folder + "/all_data.csv", index=False)
	 
 