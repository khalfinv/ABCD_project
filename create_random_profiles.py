#!/usr/bin/python3
import os, sys, argparse, pickle
import pandas as pd
from random import sample
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--subjects_file', required=True, type=str, help='path to excel file with subject keys')
	parser.add_argument('--out_folder', required=True, type=str, help='path to output folder')
	args = parser.parse_args()
	
	df = pd.read_excel(io=args.subjects_file)["SUBJECTKEY"]
	num_of_subjects = df.shape[0]
	num_class1 = 1716
	num_class2 = 2661
	num_class3 = 332
	num_class4 = 346
	all_indexes = list(range(num_of_subjects))
	#Sample class 1 indexes
	class1_indexes = sample(all_indexes, k=num_class1)
	#select the subject keys according to class 1 sample
	df_class1 = df[class1_indexes].to_frame()
	#Add column for class 1
	class1 = [1] * len(class1_indexes)
	df_class1["Class"] = class1
	#Delete from original list the sampled indexes
	all_indexes = [item for item in all_indexes if item not in class1_indexes]
	
	#Sample class 2 indexes
	class2_indexes = sample(all_indexes, k=num_class2)
	#select the subject keys according to class 2 sample
	df_class2 = df[class2_indexes].to_frame()
	#Add column for class 2
	class2 = [2] * len(class2_indexes)
	df_class2["Class"] = class2
	#Delete from original list the sampled indexes
	all_indexes = [item for item in all_indexes if item not in class2_indexes]
		
	#Sample class 3 indexes
	class3_indexes = sample(all_indexes, k=num_class3)
	#select the subject keys according to class 3 sample
	df_class3 = df[class3_indexes].to_frame()
	#Add column for class 3
	class3 = [3] * len(class3_indexes)
	df_class3["Class"] = class3
	#Delete from original list the sampled indexes
	all_indexes = [item for item in all_indexes if item not in class3_indexes]
	#select the subject keys according to indexes that left
	class4_indexes = all_indexes
	df_class4 = df[class4_indexes].to_frame()
	#Add column for class 4
	class4 = [4] * len(class4_indexes)
	df_class4["Class"] = class4
	
	#Conctinate all to one data frame and save
	all = [df_class1, df_class2, df_class3, df_class4]
	df_all = pd.concat(all)
	df_all.to_excel(args.out_folder + "/random_profiles.xlsx", index=False)
	
	print(df_class1)
	print(df_class2)
	print(df_class3)
	print(df_class4)
	print(df_all)
	
	
	