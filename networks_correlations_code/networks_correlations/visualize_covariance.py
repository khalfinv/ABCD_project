#!/usr/bin/python3
"""
==============================================================================================================
Visualize covariance between ROIs
==============================================================================================================

@Input:  
out_folder = string. Output folder
cov_mat = string. Path to covariance matrix's excel file 
vmin = float. Minimum value for scale in matrix plot
vmax = float. Maximum value for scale in matrix plot
networks = list. Choose networks for ploting. You can choose all combinations. Optional.  


@Output: 
Covariance matrices plots.
If networks input exists - Covariance matrices plots for networks and excel file of the sub covariance matrices. 
"""
from nilearn import plotting
import os, sys, pickle, argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import networkToIndexDic
import itertools
import visualize_functions as vf

	
def plotAllCombinations(common_mat, base_name, out_folder):
	"""Create plots of covariance matrices for all combinations of 2 networks. 
	param common_mat: data frame.The common matrix with all networks
	param base_name: string. Base name for the output files
	param out_folder: string. The path of output folder
	return: None
	"""
	print("Plot correlations between every two networks")
	for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
		#create a list of indexes to slice
		listToSlice = []
		network1 = (pair)[0]
		network2 = (pair)[1]
		# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix. 		
		listToSlice = listToSlice + list(networkToIndexDic.dic[network1])
		listToSlice = listToSlice + list(networkToIndexDic.dic[network2])
		common_mat_sliced = common_mat.iloc[listToSlice,listToSlice]
		#save to excel
		common_mat_sliced.to_excel(out_folder + "/" + base_name + "_" + network1 + "_" + network2 + ".xlsx") 		
		ticks = [0, len(networkToIndexDic.dic[network1]), len(networkToIndexDic.dic[network1]) + len(networkToIndexDic.dic[network2])]                                                                                                                                                                                                                                              
		vf.plotMatrix(common_mat_sliced.values, out_folder + "/" + base_name + "_" + network1 + "_" + network2 + ".png", [network1, network2],
			base_name + " - " + network1 + "_" + network2, ticks, common_mat_sliced.values.min(), common_mat_sliced.values.max())

				   
def plotSomeNetworks(common_mat, base_name, out_folder, networks, vmin, vmax):
	"""Create plots of covariance matrices for specific networks 
	param common_mat : data frame.The common matrix with all networks
	param base_name: string. Base name for the output files
	param out_folder: string. The path of output folder
	param networks: list. List of networks for matrices generating
	param vmin: float. Minimum value for scale in matrix plot
	param vmax: float. Maximum value for scale in matrix plot
	return: None
	"""
	print("Plot correlations for specified networks")
	ticks = [0]
	#create a list of indexes to slice
	listToSlice = []
	for network in networks:
		if network in networkToIndexDic.dic:
			# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
			listToSlice = listToSlice + list(networkToIndexDic.dic[network])
			ticks.append(ticks[-1] + len(networkToIndexDic.dic[network]))
		else:
			print ( "The " + network + " network does not exist!!!")
	common_mat_sliced = common_mat.iloc[listToSlice,listToSlice] 
	#save to excel
	common_mat_sliced.to_excel(out_folder + "/" + base_name + "_" + str(networks) +  ".xlsx")
	vf.plotMatrix(common_mat_sliced.values, out_folder + "/" + base_name + "_" + str(networks) + ".png", networks, base_name + " - " + str(networks),
		ticks, vmin, vmax)
		
def plotAllNetworks(common_mat,out_path, title, vmin, vmax):
	"""Create plot of covariance matrix 
	param common_mat : two dimensional array (number of parcels, number of parcels).The common correlation\covariance matrix with all networks
	param out_path: string. The path of output file
	param title: string. Plot title
	param vmin: float. Minimum value for scale in matrix plot
	param vmax: float. Maximum value for scale in matrix plot
	return: None
	"""
	print("Plot common matrix")
	ticks = [0]
	#create a list of indexes by networks order - important for not prdered atlases like Gordon
	listOfIndexes = []
	for network in networkToIndexDic.dic.keys():
		if network in networkToIndexDic.dic:
			# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
			listOfIndexes = listOfIndexes + list(networkToIndexDic.dic[network])
			ticks.append(ticks[-1] + len(networkToIndexDic.dic[network]))
		else:
			print ( "The " + network + " network does not exist!!!")
	common_mat_new = common_mat[listOfIndexes, :][:, listOfIndexes] 
	vf.plotMatrix(common_mat_new, out_path, networkToIndexDic.dic.keys(), title, ticks, vmin, vmax)
	
	
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--cov_mat', required=True, type=str, help='path to excel file containing the covariance matrix')
	parser.add_argument('--vmin', required=True, type=float, help='minimum value for scale in matrix plot')
	parser.add_argument('--vmax', required=True, type=float, help='maximum value for scale in matrix plot')
	parser.add_argument('--networks', required=False, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
	args = parser.parse_args()
	
		
	df_mat = pd.read_excel(args.cov_mat, index_col= 0)
	plotAllNetworks(df_mat.values,args.out_folder + "\covariance_matrix.png", "Common Correlation Matrix", args.vmin, args.vmax)
	
	if (args.networks != None):
		if(args.networks == ["all"]):
			plotAllCombinations(df_mat, 'covariance_matrix', args.out_folder)
		else:
			plotSomeNetworks(df_mat, 'covariance_matrix', args.out_folder,args.networks, args.vmin, args.vmax)
			