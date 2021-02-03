#!/usr/bin/python3
"""
==============================================================================================================
 
==============================================================================================================

@Input:  



@Output: 

"""



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
	parser.add_argument('--corr_mat1', required=True, type=str, help='path to excel file containing the correlation matrix 1')
	parser.add_argument('--corr_mat2', required=True, type=str, help='path to excel file containing the correlation matrix 2')
	parser.add_argument('--n1', required=True, type=int, help='Number of subjects in correlation matrix 1')
	parser.add_argument('--n2', required=True, type=int, help='Number of subjects in correlation matrix 2')
	args = parser.parse_args()
		
	df_corr_mat1 = pd.read_excel(args.corr_mat1, index_col= 0)
	df_corr_mat2 = pd.read_excel(args.corr_mat2, index_col= 0)
	z_corr_mat1 = 0.5*(np.log(1+df_corr_mat1.values)-np.log(1-df_corr_mat1.values))
	z_corr_mat2 = 0.5*(np.log(1+df_corr_mat2.values)-np.log(1-df_corr_mat2.values))
	
	x1 = 1 / (args.n1 - 3)
	x2 = 1 / (args.n2 - 3)
	z_observed = (z_corr_mat1 - z_corr_mat2) / math.sqrt(x1+x2)
	
	num_of_rois = df_corr_mat1.values.shape[0]
	#(matrix size - diagonal) / 2 (only upper/lower triangular )
	num_of_comparisons =  (num_of_rois ** 2  -  num_of_rois  ) / 2
	print("num_of_comparisons: ", num_of_comparisons)
	alpha = 0.05/num_of_comparisons #Bonferroni correction
	print("alpha: ", alpha)
	
	pd.DataFrame(z_observed).to_excel(args.out_folder + "/z_observed.xlsx")
	p_values = stats.norm.pdf(abs(z_observed))*2 #two tail 

	after_correction = p_values <= alpha #significant after bonferroni correction
	
	effect_size = abs(z_corr_mat1 - z_corr_mat2)
	pd.DataFrame(effect_size, index = df_corr_mat1.index, columns = df_corr_mat1.columns).to_excel(args.out_folder + "/effect_sizes.xlsx")
	sig_and_effect_size = np.logical_and(effect_size > 0.5 , after_correction)
	
	sig_and_es_mat_1 = df_corr_mat1[pd.DataFrame(sig_and_effect_size, index = df_corr_mat1.index, columns = df_corr_mat1.columns)]
	sig_and_es_mat_2 = df_corr_mat2[pd.DataFrame(sig_and_effect_size, index = df_corr_mat2.index, columns = df_corr_mat2.columns)]
	sig_and_es_mat_1.to_excel(args.out_folder + "/sig_and_es_mat_1.xlsx")
	sig_and_es_mat_2.to_excel(args.out_folder + "/sig_and_es_mat_2.xlsx")
	
	#Significant correlation in each matrix
	sig_values_mat_1 = df_corr_mat1[pd.DataFrame(after_correction, index = df_corr_mat1.index, columns = df_corr_mat1.columns)]
	sig_values_mat_2 = df_corr_mat2[pd.DataFrame(after_correction, index = df_corr_mat2.index, columns = df_corr_mat2.columns)]
	sig_values_mat_1.to_excel(args.out_folder + "/sig_values_mat_1.xlsx")
	sig_values_mat_2.to_excel(args.out_folder + "/sig_values_mat_2.xlsx")

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
	vf.plotMatrix(sig_values_mat_1.values, args.out_folder + "/sig_values_mat_1.png", [network], "Significant Values Matrix 1",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(sig_values_mat_1.values, coords_sliced, [network], args.out_folder, "matrix1", min_r)
		
	#plot-  significant and effect size
	vf.plotMatrix(sig_and_es_mat_1.values, args.out_folder + "/sig_and_es_mat_1.png", [network], "Significant and effect size Values Matrix 1",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(sig_and_es_mat_1.values, coords_sliced, [network], args.out_folder, "sig_and_es_matrix1", min_r)
		
	#Same for matrix 2
	#save to excel
	vf.plotMatrix(sig_values_mat_2.values, args.out_folder + "/sig_values_mat_2.png", [network], "Significant Values Matrix 2",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(sig_values_mat_2.values, coords_sliced, [network], args.out_folder, "matrix2", min_r)
		
	#plot-  significant and effect size
	vf.plotMatrix(sig_and_es_mat_2.values, args.out_folder + "/sig_and_es_mat_2.png", [network], "Significant and effect size Values Matrix 2",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(sig_and_es_mat_2.values, coords_sliced, [network], args.out_folder, "sig_and_es_matrix2", min_r)
		
		
	fig, ax = plt.subplots()
	fig.set_size_inches(16.5, 9.5)
	plt.title(label = "Significangt ROIs", fontsize = 20)
	plotting.plot_matrix(after_correction,axes = ax, cmap = 'Blues', grid = 'black')
	fig.savefig(args.out_folder + "/sig_rois.png")
	#plotting.plot_matrix(sig_values,axes = ax, grid = 'black', vmin = -1., vmax = 1.)
	
	plt.close()
	#plt.show()