from pyriemann import utils
import argparse
import pickle
import numpy as np
import pandas as pd
import nilearn
from nilearn.connectome import cov_to_corr 



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--subjects_dic', required=True, type=str, help='path to pkl file with all the subjects')
	args = parser.parse_args()
	
	pkl_file = open(args.subjects_dic, 'rb')
	allParticipantsDict = pickle.load(pkl_file)
	pkl_file.close()
	
	num_of_subjects = len(allParticipantsDict)
	print(num_of_subjects)
	
	all_mat = []
	
	for val in allParticipantsDict.values():
		all_mat.append(val["covariance"])
	
	print(np.array(all_mat).shape)
	avg_cov_mat = all_mat[0]
	for mat in all_mat[1:]:
		avg_cov_mat = np.add(avg_cov_mat,mat)
	avg_cov_mat = avg_cov_mat/num_of_subjects
	#avg_cov_mat = utils.mean.mean_covariance(np.array(all_mat),metric = 'euclid')
	avg_cor_mat = nilearn.connectome.cov_to_corr(avg_cov_mat) 
	
	df = pd.DataFrame(avg_cor_mat)
	df.to_excel(args.out_folder + "/avg_correlation_matrix.xlsx")
	df = pd.DataFrame(avg_cov_mat)
	df.to_excel(args.out_folder + "/avg_covariance_matrix.xlsx")