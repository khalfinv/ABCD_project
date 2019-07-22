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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the score')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	parser.add_argument('--networks', required=False, default=[], nargs='+', help='list of Power networks')
	parser.add_argument('--subjects_data_dict', required=True, help='path to pkl file with the subjects matrices')
	parser.add_argument('--common_data_dict', required=True, help='path to pkl file common matrices')
	args = parser.parse_args()

	#Read the excel file 
	df = pd.read_excel(io=args.score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in df):
		print("'SUBJECTKEY' column does not exists in excel file")
		sys.exit()

	if(args.score_key not in df):
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


	#change the SUBJECT_KEY values to match to SUBJECT_KEY format in the allMatDict (without "_")
	for i, row in df.iterrows(): 
		df.at[i,'SUBJECTKEY'] = re.sub(r"[^a-zA-Z0-9]","",df.at[i,'SUBJECTKEY'])

	#create dictionary where key = SUBJECT_KEY and value = {correlation matrix distance, covariance matrix distance and score} according to the Power networks(if exists in input)
	dist_to_score_dict={}
	#create a list of indexes to slice for each matrix
	listToSlice = []
	#In case of at least one network in arguments
	#Slice the common matrices according to networks coordinates
	if(len(args.networks) > 0):
		for network in args.networks:
			if network in networkToIndexDic.dic:
				listToSlice = listToSlice + list(networkToIndexDic.dic[network])
			else:
				print ( "The " + network + " network does not exist!!!")
				sys.exit()
		common_cov_mat = common_cov_mat[listToSlice, :][:, listToSlice] 
		common_cor_mat = common_cor_mat[listToSlice, :][:, listToSlice]
		
	#Calculate distance from each matrix to common matrix and save to dist_to_score_dict
	for subject_key, value in allMatDict.items():
		subject_corr_mat = value["correlation"]
		subject_cov_mat = value["covariance"]
		#Slice the subject matrix
		if(len(listToSlice) > 0):
			subject_corr_mat = subject_corr_mat[listToSlice, :][:, listToSlice]
			subject_cov_mat = subject_cov_mat[listToSlice, :][:, listToSlice]
		#calculate distance for correlation matrix and covariance matrix
		dis_corr = pyriemann.utils.distance.distance(subject_corr_mat, common_cor_mat, metric='riemann')
		dis_cov = pyriemann.utils.distance.distance(subject_cov_mat, common_cov_mat, metric='riemann')
		#subject_key = key.split("/")[-1] #The key from allMatDict is the full path to the folder. Cut only the subject_key (folder name)
		raw = df.loc[lambda df: df['SUBJECTKEY'] == subject_key] #Get the raw from the excel that match the subject_key. The raw is from type pandas.series
		dist_to_score_dict[subject_key] = {"corr_distance" : dis_corr, "cov_distance" : dis_cov, "score" : raw[args.score_key].values[0]}
		
	all_scores = [ value["score"] for value in dist_to_score_dict.values() ]
	all_corr_distances = [ value["corr_distance"] for value in dist_to_score_dict.values() ]
	all_cov_distances = [ value["cov_distance"] for value in dist_to_score_dict.values() ]
	#plot correlation graph
	fig1 = plt.figure()
	plt.title('Correlation' + str(args.networks))
	plt.xlabel('Distance')
	plt.ylabel(args.score_key + ' Score')
	plt.plot(all_corr_distances, all_scores, 'ro' )
	(corr_coef, p_value) = pearsonr(all_corr_distances, all_scores)
	plt.figtext(0.5, 0.8,"R = " + str(round(corr_coef,3)) + "    p_value = " + str(round(p_value,3)), wrap=True,
				horizontalalignment='center', fontsize=12)
	fig1.savefig("disToScoreCorr" + str(args.networks) + "_" + args.score_key + ".png")	
	print("correlation: " , corr_coef, " p_value: ", p_value)

	#plot covariance graph
	fig2 = plt.figure()
	plt.title('Covariance' + str(args.networks))
	plt.xlabel('Distance')
	plt.ylabel(args.score_key + ' Score')
	all_scores = [value["score"] for value in dist_to_score_dict.values() ]
	plt.plot(all_cov_distances, all_scores, 'ro' )
	(corr_coef, p_value) = pearsonr(all_cov_distances, all_scores)
	plt.figtext(0.5, 0.8,"R = " + str(round(corr_coef,3)) + "    p_value = " + str(round(p_value,3)), wrap=True,
				horizontalalignment='center', fontsize=12)
	fig2.savefig("disToScoreCov" + str(args.networks) + "_" + args.score_key + ".png")	
	print("covariance :" , corr_coef, " p_value: ", p_value)



	
