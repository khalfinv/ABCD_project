import argparse
#from pyriemann import utils
import pickle
import numpy as np
import pandas as pd
import nilearn
from nilearn.connectome import cov_to_corr 
from sklearn import utils



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--subjects_dic', required=True, type=str, help='path to pkl file with all the subjects')
	parser.add_argument('--sample_size', required=False, Default = 1, type=int, help='sample size ratio for bootstraping')
	parser.add_argument('--R', required=False, type=int, Default = 50, help='number of repetitions')
	args = parser.parse_args()
	
	pkl_file = open(args.subjects_dic, 'rb')
	allParticipantsDict = pickle.load(pkl_file)
	pkl_file.close()
	
	num_of_subjects = len(allParticipantsDict)
	print(num_of_subjects)
	
	all_mat = []
	networks = ['Default']
	for val in allParticipantsDict.values():
		cov_mat = val["covariance"]
		#create a list of indexes to slice
		listToSlice = []
		for network in networks:
			if network in networkToIndexDic.dic:
				# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
				listToSlice = listToSlice + list(networkToIndexDic.dic[network])
			else:
				print ( "The " + network + " network does not exist!!!")
		cov_mat_sliced = cov_mat.iloc[listToSlice,listToSlice]
		all_mat.append(cov_mat_sliced)
	
	print(len(all_mat))
	print(all_mat[10].shape)
	num_of_parcels = all_mat[0].shape[0]
	avg_cor_matrices = []
	num_of_samples = num_of_subjects * args.sample_size
	for i in range(args.R):
		data = utils.resample(all_mat, replace=True, n_samples=num_of_samples, random_state=1)
		#Calculate average
		avg_cov_mat = data[0]
		for mat in data[1:]:
			avg_cov_mat = np.add(avg_cov_mat,mat)
		avg_cov_mat = avg_cov_mat/num_of_samples
		avg_cor_mat = nilearn.connectome.cov_to_corr(avg_cov_mat)
		avg_cor_matrices.append(avg_cor_mat)		
	
	
	#Calculate confidence intervals
	alpha = 0.95
	for parcel_i in range(num_of_parcels):
		all_corr = [avg_cor_mat[parcel_i] for avg_cor_mat in avg_cor_matrices]
		print(all_corr)
		ordered = sort(all_corr)
		print(ordered)
		lower = percentile(ordered, (1-alpha)/2)
		print(lower)
		upper = percentile(ordered, alpha+((1-alpha)/2))
		print(upper)