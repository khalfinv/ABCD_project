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
covariance_matrix.xlsx: located in out_folder. Excel file with the common covariance matrix.
correlation_sum.xlsx: located in out_folder. Excel file with correlation score for between and within networks.
covariance_sum.xlsx: located in out_folder. Excel file with covariance score for between and within networks.
common_cor_matrix.png: located in out_folder. The correlation matrix plot

if networks input exists:
	sub correlation matrix plots for the networks
	brain plots (2D and 3D)
	excel files of sub correlation matrices
"""


import subprocess, argparse, os
import time
import shutil

def checkSubprocessRet(return_code):
	if (return_code != 0):
		print("Command failed!")
		exit(1)
	else:
		print("Command succeeded!")
if __name__ == "__main__":
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--preproc_folder', required=True, type=str, help='path to folder containing all the preprocessed scans')
	parser.add_argument('--atlas', required=True, type=str, help='Choose GORDON or POWER')
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--networks', required=False, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
	parser.add_argument('--min_r', required=False, type=float, default = 0.7, help='Minumun R value for brain connection plotting')
	args = parser.parse_args()
	
	current_dir = os.getcwd() #get the current working dir
	dest_dir = current_dir
	os.chdir('../Atlases')
	current_dir = os.getcwd() #get the current working dir
	atlas_dir = current_dir
	if (args.atlas == "GORDON"):
		shutil.copy(os.path.join(atlas_dir,"networkToIndexDicGordon.py"),dest_dir) #copy the file to destination dir
		dst_file = os.path.join(dest_dir,'networkToIndexDicGordon.py')
		new_dst_file_name = os.path.join(dest_dir, 'networkToIndexDic.py')
		if os.path.exists(new_dst_file_name):
			os.remove(new_dst_file_name) #remove the previous dictionary
		os.rename(dst_file, new_dst_file_name)#rename
		os.chdir(dest_dir)
		atlas_coords = os.path.join(atlas_dir,"MNI_Gordon.txt")
	elif (args.atlas == "POWER"):
		shutil.copy(os.path.join(atlas_dir,"networkToIndexDicPower.py"),dest_dir) #copy the file to destination dir
		dst_file = os.path.join(dest_dir,'networkToIndexDicPower.py')
		new_dst_file_name = os.path.join(dest_dir, 'networkToIndexDic.py')
		if os.path.exists(new_dst_file_name):
			os.remove(new_dst_file_name) #remove the previous dictionary
		os.rename(dst_file, new_dst_file_name)#rename
		os.chdir(dest_dir)
		atlas_coords = os.path.join(atlas_dir,"MNI_Power.txt")
	else:
		print (" Wrong atlas!!! Please contact if you want to add new atlas")
		
	#Command for creating the subjects dictionary pkl file
	create_subject_dict_command = "create_subjects_dict.py --preproc_folder " + args.preproc_folder + " --atlas_coords " + atlas_coords + " --out_folder " + args.out_folder
	print(create_subject_dict_command)
	return_code = subprocess.call(create_subject_dict_command, shell=True)
	checkSubprocessRet(return_code)
	#If fail - terminate the run

	subject_dict_path = args.out_folder + "/subjects_data.pkl"
	#Command for creating the common matrices
	run_common_mat_command = "run_common_mat.py --out_folder " + args.out_folder + " --subjects_data_dict " +  subject_dict_path
	print(run_common_mat_command)
	return_code = subprocess.call(run_common_mat_command, shell=True)
	checkSubprocessRet(return_code)
	#Command for plots creating
	#Correlation matrix
	visualize_corr_command = "visualize_correlation.py  --corr_mat " + args.out_folder + "\correlation_matrix.xlsx" + " --out_folder " + args.out_folder
	if args.networks != None:
		networks_str = ""
		for network in args.networks:
			networks_str = networks_str + str(network) + " "
		visualize_corr_command = visualize_corr_command + " --networks " + networks_str + " --atlas " + atlas_coords + " --min_r " + str(args.min_r)
	print(visualize_corr_command)
	return_code = subprocess.call(visualize_corr_command, shell=True)
	checkSubprocessRet(return_code)
	
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	