
"""
==============================================================================================================
Output significant fc vectors  
==============================================================================================================

"""
import numpy as np
import pandas as pd
import argparse
import pickle


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--negative_mat', required=True, type=str, help='path to excel file containing the negative significant correlation matrix')
	parser.add_argument('--positive_mat', required=True, type=str, help='path to excel file containing the positive significant correlation matrix')
	parser.add_argument('--subjects_data', required=True, help='path to pkl file containing the subjects correlation matrices')
	args = parser.parse_args()
	
	negative_mat = pd.read_excel(args.negative_mat, index_col= 0).to_numpy()
	positive_mat = pd.read_excel(args.positive_mat, index_col= 0).to_numpy()
	pkl_file = open(args.subjects_data, 'rb')
	allParticipantsDict = pickle.load(pkl_file)
	pkl_file.close()
	
	for val in allParticipantsDict.values():
		corr_mat = val["correlation"].to_numpy()
		sig_fc = (df_negative+df_positive)*corr_mat
		print (sig_fc)
		break
		
	
	