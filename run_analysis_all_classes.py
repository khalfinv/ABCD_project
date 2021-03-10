import os, sys, argparse, pickle
import pandas as pd
import subprocess
import time

def checkSubprocessRet(return_code):
	if (return_code != 0):
		print("Command failed!")
		exit(1)
	else:
		print("Command succeeded!")
		
		
if __name__ == "__main__":
	start = time.time()
	# common_mat_commands = [r"networks_correlations_code\\networks_correlations\\run_common_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class1 --subjects_data_dict Y:\Vicki\TwoGroups\Version1\Group2\Class1\subjects_data_class_1.pkl"
	# ,"networks_correlations_code\\networks_correlations\\run_common_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class2 --subjects_data_dict Y:\Vicki\TwoGroups\Version1\Group2\Class2\subjects_data_class_2.pkl"
	# ,"networks_correlations_code\\networks_correlations\\run_common_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class3 --subjects_data_dict Y:\Vicki\TwoGroups\Version1\Group2\Class3\subjects_data_class_3.pkl"
	# ,"networks_correlations_code\\networks_correlations\\run_common_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class4 --subjects_data_dict Y:\Vicki\TwoGroups\Version1\Group2\Class4\subjects_data_class_4.pkl"
	# ]
	
	# for command in common_mat_commands:
		# print(command)
		# return_code = subprocess.call(command, shell=True)
		# checkSubprocessRet(return_code)
		
	# visualize_correlation_commands = ["networks_correlations_code\\networks_correlations\\visualize_correlation.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class1 --corr_mat Y:\Vicki\TwoGroups\Version1\Group2\Class1\correlation_matrix.xlsx --networks Default --atlas networks_correlations_code\Atlases\MNI_Gordon.txt"
	# , "networks_correlations_code\\networks_correlations\\visualize_correlation.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class2 --corr_mat Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix.xlsx --networks Default --atlas networks_correlations_code\Atlases\MNI_Gordon.txt"
	# , "networks_correlations_code\\networks_correlations\\visualize_correlation.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class3 --corr_mat Y:\Vicki\TwoGroups\Version1\Group2\Class3\correlation_matrix.xlsx --networks Default --atlas networks_correlations_code\Atlases\MNI_Gordon.txt"
	# , "networks_correlations_code\\networks_correlations\\visualize_correlation.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class4 --corr_mat Y:\Vicki\TwoGroups\Version1\Group2\Class4\correlation_matrix.xlsx --networks Default --atlas networks_correlations_code\Atlases\MNI_Gordon.txt"]
	
	# for command in visualize_correlation_commands:
		# print(command)
		# return_code = subprocess.call(command, shell=True)
		# checkSubprocessRet(return_code)
		
	significant_ROIs_commands = ["networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class1-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class1\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 882 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt"
    , "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class1-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class1\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 882 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt --thr 0.2"
    , "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class1-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class1\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 882 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt --thr 0.1"
	, "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class3-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class3\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 161 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt"
    , "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class3-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class3\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 161 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt --thr 0.2"
    , "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class3-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class3\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 161 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt --thr 0.1"
	, "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class4-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class4\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 170 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt"
    , "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class4-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class4\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 170 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt --thr 0.2"
    , "networks_correlations_code\\networks_correlations\\statistics\\significant_ROIs.py --out_folder Y:\Vicki\TwoGroups\Version1\Group2\Class4-Class2 --corr_mat1 Y:\Vicki\TwoGroups\Version1\Group2\Class4\correlation_matrix_['Default'].xlsx --corr_mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class2\correlation_matrix_['Default'].xlsx --n1 170 --n2 1314 --atlas networks_correlations_code\Atlases\MNI_Gordon.txt --thr 0.1"]
	
	for command in significant_ROIs_commands:
		print(command)
		return_code = subprocess.call(command, shell=True)
		checkSubprocessRet(return_code)
        
        
    
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	