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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--X_files', required=True, nargs='+', type=str, help='path to excel files containing the examples')
	parser.add_argument('--Y_file', required=True, type=str, help='path to excel file containing the labels')
	parser.add_argument('--X_columns', required=True,  nargs='+', type=str, help='name of column for X examples')
	parser.add_argument('--Y_column', required=True, type=str, help='name of the column for Y labels')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()


	X_df_list = []
	X_df = pd.read_excel(io=args.X_files[0])
	#Read the X excel files
	for file_path in args.X_files[1:]:
		df = pd.read_excel(io=file_path)
		X_df = pd.merge(left=X_df, right=df,  how='inner', left_on='SUBJECTKEY', right_on='SUBJECTKEY')
	X_df = X_df[['SUBJECTKEY'] + args.X_columns]
	#Read the Y excel file
	y_df = pd.read_excel(io=args.Y_file)
	y_df = y_df[['SUBJECTKEY'] + [args.Y_column]]
	X_y_df = pd.merge(left=X_df, right=y_df, left_on='SUBJECTKEY', right_on='SUBJECTKEY').dropna()
	X_y_df.rename(columns={args.Y_column:'label'}, inplace=True)
	
	#Save the data
	X_y_df.to_csv(path_or_buf=args.out_folder + "/all_data.csv", index=False)
	 
 