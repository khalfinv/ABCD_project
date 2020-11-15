#!/usr/bin/python3
import os, sys, argparse, pickle
	
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

	for key,val in allParticipantsDict.items():
		print((df_profiles[df_profiles["SUBJECTKEY"] == key]["Class"])