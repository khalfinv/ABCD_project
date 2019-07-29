#!/usr/bin/python3
"""
==========================================================================================
Go over the subjects_data dictionary after post processing and locate all the subjects with
number of volumes < exclusion_criteria 
==========================================================================================

@Input:  
out_folder = string. Output folder for excluded_subject.txt file
exclusion_criteria = integer. Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)
subjects_data_dict = string. Path to pkl file containing the subjects' data dictionary.

@Output:
excluded_subject.txt file: located in out_folder. This file contains subjects' keys, that were excluded from analysis. 	

"""
import os, sys, pickle, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', required=True, type=str, help='output folder for excluded_subject.txt file')
    parser.add_argument('--exclusion_criteria', required=False, type=int, default=375, help='Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)')
    parser.add_argument('--subjects_data_dict', required=True, type=str, help='path to pkl file containing the subjects data dictionary')
    args = parser.parse_args()
   
    #open the output file
    excluded_subjects_file = open(args.out_folder + "/excluded_subjects.txt", 'w')
    #read the subject_data dictionary
    pkl_file = open(args.subjects_data_dict, 'rb')
    allParticipantsDic = pickle.load(pkl_file)
    pkl_file.close()
    for key,val in allParticipantsDic.items():
        if val["num_of_volumes"] < args.exclusion_criteria:
            excluded_subjects_file.write(key + "\n")
    excluded_subjects_file.close()  