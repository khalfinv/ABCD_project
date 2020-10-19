"""
==================================================================================================
Creates the following graphs:
- Subject's distance from his correlation matrix to common correlation matrix and behavioral score
- Subject's distance from his covariance matrix to common covarianace matrix and behavioral score
The distance can be calculated to some Power networks or for all the networks
==================================================================================================

@Input:  
score_file = path to excel file with the wanted test score
score_key = the column name of the test score
networks = networks for slicing and distance calculation. Not mandatory.  
subjects_data_dict = path to pkl file with the subjects matrices
common_data_dict = path to pkl file common matrices
out_folder = path to output folder. Not mandatory, the default is the current folder
exclusion_criteria = Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)

@Output:
Two graphs - correlation and covariance matrices distance vs the behavioral test score. The R value and p_value are plotted on the graph.  
"""

import os, sys, re, pickle, argparse
import pandas as pd
import pyriemann
import matplotlib.pyplot as plt
import numpy as np
import networkToIndexDic 
from scipy.stats.stats import pearsonr

def score_vs_distance(networks,common_cov_mat,common_cor_mat,allMatDict, scores_file, score_key, exclusion_criteria = 375) :  
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
	#create dictionary where key = SUBJECT_KEY and value = {correlation matrix distance, covariance matrix distance and score} according to the Power networks(if exists in input)
	dist_to_score_dict={}
	#create a list of indexes to slice for each matrix
	listToSlice = []
	#In case of at least one network in arguments
	#Slice the common matrices according to networks coordinates
	if(len(networks) > 0):
		print (', '.join(networks))
		for network in networks:
			if network in networkToIndexDic.dic:
				listToSlice = listToSlice + list(networkToIndexDic.dic[network])
			else:
				print ( "The " + network + " network does not exist!!!")
				sys.exit()
		common_cov_mat = common_cov_mat[listToSlice, :][:, listToSlice] 
		common_cor_mat = common_cor_mat[listToSlice, :][:, listToSlice]
		
	#Calculate distance from each matrix to common matrix and save to dist_to_score_dict
	for subject_key, value in allMatDict.items():
		if (value["num_of_volumes"] >= exclusion_criteria):
			subject_corr_mat = value["correlation"]
			subject_cov_mat = value["covariance"]
			#Slice the subject matrix
			if(len(listToSlice) > 0):
				subject_corr_mat = subject_corr_mat[listToSlice, :][:, listToSlice]
				subject_cov_mat = subject_cov_mat[listToSlice, :][:, listToSlice]
			#calculate distance for correlation matrix and covariance matrix
			dis_corr = pyriemann.utils.distance.distance(subject_corr_mat, common_cor_mat, metric='riemann')
			dis_cov = pyriemann.utils.distance.distance(subject_cov_mat, common_cov_mat, metric='riemann')
			subject_key = subject_key.split("NDAR")
			subject_key = "NDAR_" + subject_key[1]
			raw = scores_file.loc[lambda scores_file: scores_file['SUBJECTKEY'] == subject_key] #Get the raw from the excel that match the subject_key. The raw is from type pandas.series
			if raw.empty == False:
				dist_to_score_dict[subject_key] = {"corr_distance" : dis_corr, "cov_distance" : dis_cov, "score" : raw[score_key].values[0]}
			else:
				print(subject_key, "not found")
			
	all_scores = []
	all_corr_distances = []
	all_cov_distances = []
	for value in dist_to_score_dict.values():
		all_scores.append(value["score"])
		all_corr_distances.append(value["corr_distance"])
		all_cov_distances.append(value["cov_distance"])
	
	return all_scores, all_cov_distances, all_corr_distances
	
	
def calc_correlation(networks, score_key, all_cov_distances, all_corr_distances, all_scores, out_folder):
	"""Calculates correlation and p-value between distances and behavioural scores and plots the graphs.  
	param networks: list. 
	param score_key: string. The behavioural score name
	param all_cov_distances : list. Distances from subjects' covariance matrix to common matrix 
	param all_corr_distances : list. Distances from subjects' correlation matrix to common matrix 
	param all_scores : list. Subjects' behavioural scores
	param out_folder : string. Output folder path

	return: None

	"""  
	
	#calc pearson correlation and plot correlation distance vs score graph
	fig1 = plt.figure()
	plt.title('Correlation' + str(networks))
	plt.xlabel('Distance')
	plt.ylabel(score_key + ' Score')
	plt.plot(all_corr_distances, all_scores, 'ro' )
	(corr_coef, p_value) = pearsonr(all_corr_distances, all_scores)
	if (p_value < 0.05):
		print("Significance was found!!!")
	plt.figtext(0.5, 0.8,"R = " + str(round(corr_coef,3)) + "    p_value = " + str(round(p_value,3)), wrap=True,
				horizontalalignment='center', fontsize=12)
	fig1.savefig(out_folder + "\disToScoreCorr" + str(networks) + "_" + score_key + ".png")	
	print("correlation: " , corr_coef, " p_value: ", p_value)

	#calc pearson correlation and plot covariance distance vs score  graph
	fig2 = plt.figure()
	plt.title('Covariance' + str(networks))
	plt.xlabel('Distance')
	plt.ylabel(score_key + ' Score')
	plt.plot(all_cov_distances, all_scores, 'ro' )
	(corr_coef, p_value) = pearsonr(all_cov_distances, all_scores)
	if (p_value < 0.05):
		print("Significance was found!!!")
	plt.figtext(0.5, 0.8,"R = " + str(round(corr_coef,3)) + "    p_value = " + str(round(p_value,3)), wrap=True,
				horizontalalignment='center', fontsize=12)
	fig2.savefig(out_folder + "\disToScoreCov" + str(networks) + "_" + score_key + ".png")	
	print("covariance :" , corr_coef, " p_value: ", p_value)
	plt.close('all')

	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the score')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	parser.add_argument('--networks', required=False, default=[], nargs='+', help='list of Power networks')
	parser.add_argument('--subjects_data_dict', required=True, help='path to pkl file with the subjects matrices')
	parser.add_argument('--common_data_dict', required=True, help='path to pkl file common matrices')
	parser.add_argument('--out_folder', required=False, default='.', help='path to output folder')
	parser.add_argument('--exclusion_criteria', required=False, type=int, default=375, help='Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)')
	args = parser.parse_args()

	#Read the excel file 
	scores_file = pd.read_excel(io=args.score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in scores_file):
		print("'SUBJECTKEY' column does not exists in excel file")
		sys.exit()

	if(args.score_key not in scores_file):
		print(args.score_key, " column does not exist in excel file")
		sys.exit()
		
	#open and read the pkl file of all subjects' matrices	
	pkl_file = open(args.subjects_data_dict, 'rb')
	allMatDict = pickle.load(pkl_file)
	pkl_file.close()

	#open and read the pkl file of the common matrices	
	pkl_file = open(args.common_data_dict, 'rb')
	(common_cov_mat, common_cor_mat) = pickle.load(pkl_file)
	pkl_file.close()

	all_scores, all_cov_distances, all_corr_distances  = score_vs_distance(args.networks, common_cov_mat, common_cor_mat, allMatDict, scores_file, args.score_key, args.exclusion_criteria)   
	calc_correlation(args.networks, args.score_key, all_cov_distances, all_corr_distances, all_scores, args.out_folder)



	
