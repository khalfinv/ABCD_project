import argparse
import os
import pickle
from common_statistics import est_common_cov
import numpy as np
import nilearn
from nilearn.connectome import cov_to_corr 
from sklearn import utils
import networks_correlations.networkToIndexDic as net_dic
import pandas as pd



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--subjects_dic', required=True, type=str, help='path to pkl file with all the subjects')
	parser.add_argument('--sample_size', required=False, default = 1, type=float, help='sample size ratio for bootstraping')
	parser.add_argument('--R', required=False, type=int, default = 50, help='number of repetitions')
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
			if network in net_dic.dic:
				# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
				listToSlice = listToSlice + list(net_dic.dic[network])
			else:
				print ( "The " + network + " network does not exist!!!")
		cov_mat_sliced = cov_mat[listToSlice, :]
		cov_mat_sliced = cov_mat_sliced[: , listToSlice]
		all_mat.append(cov_mat_sliced)
	
	print(all_mat[10].shape)
	num_of_parcels = all_mat[0].shape[0]
	common_cor_matrices = []
	num_of_samples = int(num_of_subjects * args.sample_size)
	print(num_of_samples)
	for i in range(args.R):
		data = utils.resample(all_mat, replace=True, n_samples=num_of_samples)
		#Calculate average
		# avg_cov_mat = data[0]
		# for mat in data[1:]:
			# avg_cov_mat = np.add(avg_cov_mat,mat)
		# avg_cov_mat = avg_cov_mat/num_of_samples
		common_cov_mat = est_common_cov(data)
		common_cor_mat = nilearn.connectome.cov_to_corr(common_cov_mat)
		common_cor_matrices.append(common_cor_mat)		
	
	
	#Calculate confidence intervals
	alpha = 0.25
	all_CI = np.empty((num_of_parcels,num_of_parcels), dtype = list)
	for parcel_i in range(num_of_parcels):
		for parcel_j in range(num_of_parcels):
			all_corr = [common_cor_mat[parcel_i,parcel_j] for common_cor_mat in common_cor_matrices]
			print(all_corr)
			all_corr.sort()
			print(all_corr)
			lower = np.percentile(all_corr, (alpha/2)*100, interpolation = 'nearest')
			print(lower)
			upper = np.percentile(all_corr, (1-(alpha/2))*100, interpolation = 'nearest')
			print(upper)
			all_CI[parcel_i,parcel_j] = [lower,upper]
	print(all_CI)
	with open(os.path.join(args.out_folder,'CI.npy'), 'wb') as f:
		np.save(f, all_CI)
		
	pd.DataFrame(all_CI).to_excel(os.path.join(args.out_folder,'CI.xlsx'),index=False)