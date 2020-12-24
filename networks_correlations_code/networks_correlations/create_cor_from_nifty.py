#!/usr/bin/python3
"""
==============================================================================================================
Create common covariance and correlation matrix according to Zhitnikov et al. from preprocessed nifty file.
==============================================================================================================

@Input:  
preproc_folder = string. Path to folder conatining all the preprocessed scans
atlas_coords = string. path to text file with all the atlas coordinates
out_folder = string. Path to output folder
networks = list. Plot the common matrices for some networks, if you want the combination of every two networks use --networks all
min_r = float. Minimal R value (0-1) for brain connections plotting. The default is 0.7. 


@Output: 
correlation_matrix.xlsx: located in out_folder. Excel file with the common correlation matrix.
covariance_matrix.xlsx: located in out_folder. Excel file with the common correlation matrix.
correlation_sum.xlsx: located in out_folder. Excel file with correlation score for between and within networks.
covariance_sum.xlsx: located in out_folder. Excel file with covariance score for between and within networks.
common_cor_matrix.png: located in out_folder. The correlation matrix plot
common_cov_matrix.png: located in out_folder. The covariance matrix plot
if networks input exists:
	sub covariance and correlations matrices plots for the networks
	brain plots (2D and 3D)
	excel files of sub covariance and correlation matrices
"""


import subprocess, argparse, os
import time


if __name__ == "__main__":
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--preproc_folder', required=True, type=str, help='path to folder containing all the preprocessed scans')
	parser.add_argument('--atlas_coords', required=True, type=str, help='path to text file with all the atlas coordinates')
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--networks', required=False, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
	parser.add_argument('--min_r', required=False, type=float, default = 0.7, help='Minumun R value for brain connection plotting')
	args = parser.parse_args()
	#Command for creating the subjects dictionary pkl file
	create_subject_dict_command = "create_subjects_dict.py --preproc_folder " + args.preproc_folder + " --atlas_coords " + args.atlas_coords + " --out_folder " + args.out_folder
	print(create_subject_dict_command)
	subprocess.call(create_subject_dict_command, shell=True)
	subject_dict_path = args.out_folder + "/subjects_data.pkl"
	#Command for creating the common matrices
	run_common_mat_command = "run_common_mat.py --out_folder " + args.out_folder + " --subjects_data_dict " +  subject_dict_path
	print(run_common_mat_command)
	#Command for plots creating
	visualize_command = "visualize.py  --corr_mat " + args.out_folder + "\correlation_matrix.xlsx" + " --cov_mat " + args.out_folder + "\covariance_matrix.xlsx" + " --out_folder " + args.out_folder
	if args.networks != None:
		networks_str = ""
		for network in args.networks:
			networks_str = networks_str + str(network) + " "
		visualize_command = visualize_command + " --networks " + networks_str + " --atlas " + args.atlas_coords + " --min_r " + str(args.min_r)
	print(visualize_command)
	subprocess.call(visualize_command, shell=True)
	
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	