
import os, sys, pickle, argparse
import networkToIndexDic
import itertools
import pandas as pd
from scipy.stats.stats import pearsonr
import numpy as np 


def score_vs_corr(network1,network2,common_cor_mat,allMatDict, scores_file, score_key, common_mean_corr, within = True, exclusion_criteria = 375) :  
	"""Calculate riemann distance from each subject's matrix to common matrix and match to behavioural test score 
	param networks: list. Networks for slicing and distance calculation, can be empty.
	param common_cov_mat : two dimentional array . Common covariance matrix of size (264, 264)
	param common_cor_mat : two dimentional array. Common correlation matrix of size (264, 264)
	param allMatDict: dictionary. all subject's data. 
								  key: SUBJECT_KEY 
								  value: {"time_series" : matrix of time series after censoring (time_points, power_rois), "covariance" : covariance matrix of power rois (power_rois, power_rois),
															"correlation" : correlation matrix of power rois (power_rois, power_rois), "num_of_volumes" : num of volumes left after censoring
	param scores_file: DataFrame . File with subject's behavioural scores and data
	param score_key : string. The wanted score's column name in scores_file
	param exclusion_criteria : integer. Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)
	return: 3 lists of the same length were each index is correspond to some subject, accordingly
		all_scores: list of subjects's scores
		all_cov_distances: list of subjects' covariance matrices distances
		all_corr_distances: list of subjects' correlation matrices distances

	"""  
	#create dictionary where key = SUBJECT_KEY and value = {fc correlation difference and score} 
	dist_to_score_dict={}

	for subject_key, value in allMatDict.items():
		if (value["num_of_volumes"] >= exclusion_criteria):
			subject_corr_mat = value["correlation"]
			r_sum = 0
			count = 0
			if (within == False):
				for i in networkToIndexDic.dic[network1]:
					for j in networkToIndexDic.dic[network2]:
						r_sum = r_sum + subject_corr_mat[i][j]
						count = count + 1
			else:
				start_index = networkToIndexDic.dic[network1][0]
				end_index = networkToIndexDic.dic[network1][-1] + 1
				for i in range(start_index + 1, end_index) :
					for j in range(start_index, i):
						r_sum = r_sum + subject_corr_mat[i][j]
						count = count + 1
			mean_corr = r_sum/count
			subject_key = subject_key.split("NDAR")
			subject_key = "NDAR_" + subject_key[1]
			raw = scores_file.loc[lambda scores_file: scores_file['SUBJECTKEY'] == subject_key] #Get the raw from the excel that match the subject_key. The raw is from type pandas.series
			dist_to_score_dict[subject_key] = {"corr_diff" : abs(mean_corr - common_mean_corr), "score" : raw[score_key].values[0]}
			
	all_scores = []
	all_corr_diff = []
	for value in dist_to_score_dict.values():
		all_scores.append(value["score"])
		all_corr_diff.append(value["corr_diff"])
	
	return all_scores, all_corr_diff

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--common_mat_pkl', required=True, type=str, help='path to common matrices pkl file')
	parser.add_argument('--subjects_data_dict', required=True, help='path to pkl file containing the subjects data dictionary')
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the score')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	args = parser.parse_args()
	
	#use previous version of common matrices
	print("Read from common matrices pkl file")
	pkl_file = open(args.common_mat_pkl, 'rb')
	(common_cov_mat, common_cor_mat) = pickle.load(pkl_file)
	pkl_file.close()
	
	#open and read the pkl file of all subjects' matrices	
	pkl_file = open(args.subjects_data_dict, 'rb')
	allMatDict = pickle.load(pkl_file)
	pkl_file.close()
	
	#Read the excel file 
	scores_file = pd.read_excel(io=args.score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in scores_file):
		print("'SUBJECTKEY' column does not exists in excel file")
		sys.exit()

	if(args.score_key not in scores_file):
		print(args.score_key, " column does not exist in excel file")
		sys.exit()
		
	# #convert to Z-score
	# for i in range(len(common_cor_mat)):
		# for j in range(len(common_cor_mat[i])):
			# common_cor_mat[i][j] = np.arctanh(common_cor_mat[i][j])
	
	
	#between
	for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
		#create a list of indexes to slice
		listToSlice = []
		network1 = (pair)[0]
		network2 = (pair)[1]
		print (network1, network2)
		r_sum = 0
		count = 0
		for i in networkToIndexDic.dic[network1]:
			for j in networkToIndexDic.dic[network2]:
				r_sum = r_sum + common_cor_mat[i][j]
				count = count + 1
		mean_corr = r_sum / count
		print ("r_sum: ", r_sum, " count: ", count, " mean: ", mean_corr)
		(all_scores, all_corr_diff) = score_vs_corr(network1, network2, common_cor_mat, allMatDict, scores_file, args.score_key, mean_corr, within = False)
		(corr_coef, p_value) = pearsonr(all_corr_diff, all_scores)
		print ("corr_coef score vs distance: ", corr_coef, " p_value: ", p_value)
	#within 
	for network in networkToIndexDic.dic.keys():
		r_sum = 0
		count = 0
		print (network)
		start_index = networkToIndexDic.dic[network][0]
		end_index = networkToIndexDic.dic[network][-1] + 1
		for i in range(start_index + 1, end_index) :
			for j in range(start_index, i):
				r_sum = r_sum + common_cor_mat[i][j]
				count = count + 1
		mean_corr = r_sum / count
		print ("r_sum: ", r_sum, " count: ", count, " mean: ", mean_corr)
		(all_scores, all_corr_diff) = score_vs_corr(network, network, common_cor_mat, allMatDict, scores_file, args.score_key, mean_corr, within = True)
		(corr_coef, p_value) = pearsonr(all_corr_diff, all_scores)
		if (corr_coef > 0.1):
			print ("big r found!!!")
		print ("corr_coef score vs distance: ", corr_coef, " p_value: ", p_value)
			