#!/usr/bin/python3
"""
==============================================================================================================
Create common covariance and correlation matrix according to Zhitnikov et al. 
==============================================================================================================

@Input:  
out_folder = string. Output folder for excel files and plots
subjects_data_dict = string. Path to pkl file containing the subjects' data dictionary


@Output: 
correlation_matrix.xlsx: located in out_folder. Excel file with the common correlation matrix.
correlation_sum.xlsx: located in out_folder. Excel file with correlation score for between and within networks.
		

"""

import os, sys, pickle, argparse
from common_statistics import snr, est_common_cov, est_common_density2D
from nilearn.connectome import cov_to_corr 
import nilearn
import time
import networkToIndexDic
import pandas as pd
import itertools



def createCommonMat(subjects_data_dict):
	"""Create common covariance and correlation matrices
	param subjects_data_dict: dictionary. All subject's data dictionary from post processing step
	return: (common_cov_mat, common_cor_mat) : tuple. 
		common_cov_mat - common covariance matrix (264, 264), common_cor_mat - common correlation matrix (264, 264)
	"""
	print("Create common correlation matrix")
	#covariance matrices
	covars = []
	for val in subjects_data_dict.values():
		covars.append(val["covariance"])
	print("number of subjects: ", len(covars))
	#find common covariance matrix
	common_cov_mat = est_common_cov(covars)
	#create common correlation matrix
	common_cor_mat = nilearn.connectome.cov_to_corr(common_cov_mat)
	return (common_cov_mat, common_cor_mat)



def sumCorrScore(out_folder, common_cor_mat):
	"""Sun correlation scores for within and between networks and save the results to excel file. 
	param out_folder: string. Output folder for excel file
	param common_cor_mat : two dimensional array. The correlation matrix
	return: None
	"""
	df_raws = []
	#Calculate between correlations
	print("Calculating between correlations")
	for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
		new_raw = {}
		#create a list of indexes to slice
		listToSlice = []
		network1 = (pair)[0]
		network2 = (pair)[1]
		r_sum = 0
		count = 0
		for i in networkToIndexDic.dic[network1]:
			for j in networkToIndexDic.dic[network2]:
				r_sum = r_sum + common_cor_mat[i][j]
				count = count + 1
		mean_corr = r_sum / count
		new_raw['networks name'] = network1 + '_' + network2
		new_raw['correlation score'] = mean_corr
		df_raws.append(new_raw)
		#print ("r_sum: ", r_sum, " count: ", count, " mean: ", mean_corr)
		
	#Calculate within correlations 
	print("Calculating within correlations")
	for network in networkToIndexDic.dic.keys():
		new_raw = {}
		r_sum = 0
		count = 0
		parcels = networkToIndexDic.dic[network]
		num_of_parcels = len(networkToIndexDic.dic[network])
		for i in range(num_of_parcels):
			for j in range(i+1,num_of_parcels):
				r_sum = r_sum + common_cor_mat[parcels[i]][parcels[j]]
				count = count + 1
		if (num_of_parcels > 1):
			mean_corr = r_sum / count
			new_raw['networks name'] = network + '_' + network
			new_raw['correlation score'] = mean_corr
			df_raws.append(new_raw)
		#print ("r_sum: ", r_sum, " count: ", count, " mean: ", mean_corr)
		
	#Save to excel file
	df = pd.DataFrame(df_raws) 
	df.to_excel(out_folder + "/correlation_sum.xlsx", index=False)

	

if __name__ == "__main__":
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder for common matrices pkl file (if generated) and plots')
	parser.add_argument('--subjects_data_dict', required=True, type=str, help='path to pkl file containing the subjects data dictionary')
	args = parser.parse_args()
	

	# generated new common covariance and correlation matrices
	pkl_file = open(args.subjects_data_dict, 'rb')
	allParticipantsDict = pickle.load(pkl_file)
	pkl_file.close()
	(common_cov_mat, common_cor_mat) = createCommonMat(allParticipantsDict)
	num_of_parcels = common_cor_mat.shape[0]
	#Save common correlation to excel
	columns = [""] * num_of_parcels
	for network,indexes in networkToIndexDic.dic.items():
		for i in indexes:
			columns[i] = network
	df = pd.DataFrame(common_cor_mat, columns = columns,  index=columns)
	df.to_excel(args.out_folder + "/correlation_matrix.xlsx")
	
	#Summarize whitin and detween correlation
	sumCorrScore(args.out_folder, common_cor_mat)
	
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("run_common_mat exucation time")
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	