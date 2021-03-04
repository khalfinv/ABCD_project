from scipy import stats
import os, sys, argparse
import pandas as pd
import math
from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np
import networks_correlations.visualize_functions as vf
import networks_correlations.networkToIndexDic as net_dic


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--corr_mat', required=True, type=str, help='path to excel file containing the correlation matrix')
	parser.add_argument('--diff_ROIs', required=True, type=str, help='path to npy file containing the boolean values for each ROI')
	args = parser.parse_args()
	
	
	df_corr_mat = pd.read_excel(args.corr_mat, index_col= 0)
	with open(args.diff_ROIs, 'rb') as diff_ROIs_f:
		diff_ROIs = np.load(diff_ROIs_f,allow_pickle=True)
	df_corr_mat = df_corr_mat[pd.DataFrame(diff_ROIs, index = df_corr_mat.index, columns = df_corr_mat.columns)]
	df_corr_mat.to_excel(os.path.join(args.out_folder,'corr_mat.xlsx'))
	
	#Visualization- need to modify
	#extract coordinates from atlas
	mniCoordsFile = open("../../Atlases/MNI_Gordon.txt","rb")
	coords = []
	for line in mniCoordsFile.read().splitlines():
		splitedLine = line.decode().split()
		newCoord = []
		for part in splitedLine:
			if part != '':
				newCoord.append(float(part))
		coords.append(newCoord)
	mniCoordsFile.close()
	
	#Slice DMN
	ticks = [0]
	min_r = 0
	#create a list of indexes to slice
	listToSlice = []
	network = "Default"
	# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
	listToSlice = listToSlice + list(net_dic.dic[network])
	ticks.append(ticks[-1] + len(net_dic.dic[network]))
	#plot- only significant
	vf.plotMatrix(df_corr_mat.values, os.path.join(args.out_folder,"corr_mat.png"), [network], "Different Values",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(df_corr_mat.values, coords_sliced, [network], args.out_folder, "diff_ROIs", min_r)