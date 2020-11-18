#!/usr/bin/python3
import os, sys, argparse, pickle
import pandas as pd
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--subjects_dic', required=True, type=str, help='path to okl file with all subjects')
	parser.add_argument('--profiles', required=True, type=str, help='path to excel file with the profiles')
	parser.add_argument('--out_folder', required=True, type=str, help='path to output folder')
	args = parser.parse_args()

	df_profiles = pd.read_excel(io=args.profiles)[["Class","SUBJECTKEY"]]
	print(df_profiles)
	pkl_file = open(args.subjects_dic, 'rb')
	allParticipantsDict = pickle.load(pkl_file)
	pkl_file.close()
	
	all_classes = df_profiles["Class"].unique()
	all_classes_dic = {}
	for c in all_classes:
		all_classes_dic[c] = {}
	for key,val in allParticipantsDict.items():
		subject_key = key
		#subject_key = subject_key.split('-')[1]
		#subject_key = subject_key[:4] + '_' + subject_key[4:]
		subject_row = df_profiles[df_profiles["SUBJECTKEY"] == subject_key]
		if(subject_row.empty == False):
			c = subject_row["Class"].values[0]
			all_classes_dic[c][subject_key] = val
		
	for c in all_classes:
		f = open(args.out_folder + '/subjects_data_class_' + str(c) +'.pkl', mode="wb")
		pickle.dump(all_classes_dic[c], f)
		f.close()
	