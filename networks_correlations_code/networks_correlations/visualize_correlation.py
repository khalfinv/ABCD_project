#!/usr/bin/python3
"""
==============================================================================================================
Visualize correlation between ROIs
==============================================================================================================

@Input:  
out_folder = string. Output folder
corr_mat = string. Path to correlation matrix's excel file 
networks = list. Choose networks for ploting. You can choose all combinations. Optional.  
atlas = string. Path to text file with the atlas' coordinates. Required if networks exists.
min_r = float. Minimal R value (0-1) for brain connections plotting. The default is 0.7. 


@Output: 
Correlation matrix plot.
If networks input exists - Correlation matrices plots for networks, brain plots and excel file of the sub correlation matrices. 
"""
from nilearn import plotting
import os, sys, pickle, argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import visualize_functions as vf
import networkToIndexDic
import itertools

	
def plotAllCombinations(common_mat, base_name, out_folder, coords, min_r):
	"""Create plots of covariance\correlation matrices for all combinations of 2 networks. 
	param common_mat: data frame.The common matrix with all networks
	param base_name: string. Base name for the output files
	param out_folder: string. The path of output folder
	param coords: list.List of atlas coordinates
	param min_r: float. The minimun R value for connection plotting.
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
			base_name + " - " + network1 + "_" + network2, ticks)
		#Plot brain connectome if all coordinates exists for those networks
		if(max(listToSlice) < len(coords)):
			coords_sliced = [coords[i] for i in listToSlice]
			vf.plotConnectome(common_mat_sliced.values, coords_sliced, pair, out_folder, base_name, min_r)

				   
def plotSomeNetworks(common_mat, base_name, out_folder, networks, coords, min_r):
	"""Create plots of covariance\correlation matrices for specific networks 
	param common_mat : data frame.The common matrix with all networks
	param base_name: string. Base name for the output files
	param out_folder: string. The path of output folder
	param networks: list. List of networks for matrices generating
	param min_r: float. The minimun R value for connection plotting.
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
		ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(common_mat_sliced.values, coords_sliced, networks, out_folder, base_name, min_r)
		
def plotAllNetworks(common_mat,out_path, title):
	"""Create plot of covariance\correlation matrix 
	param common_mat : two dimensional array (number of parcels, number of parcels).The common correlation\covariance matrix with all networks
	param out_path: string. The path of output file
	param title: string. Plot title
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
	vf.plotMatrix(common_mat_new, out_path, networkToIndexDic.dic.keys(), title, ticks)
	
	
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--corr_mat', required=True, type=str, help='path to excel file containing the correlation matrix')
	parser.add_argument('--networks', required=False, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
	parser.add_argument('--atlas', required=False, help='path to atlas coordinates file. Needed only if --networks flag exists')
	parser.add_argument('--min_r', required=False, type=float, default = 0.7, help='Minumun R value for brain connection plotting')
	args = parser.parse_args()
	
	#Arguments checks
	if (args.networks != None and args.atlas is None):
		print ("--atlas is required. Use -h for more details")
		exit(1)
		
	df_corr_mat = pd.read_excel(args.corr_mat, index_col= 0)
	plotAllNetworks(df_corr_mat.values,args.out_folder + "\correlation_matrix.png", "Common Correlation Matrix")
	
	if args.atlas != None:
		#extract coordinates from atlas
		mniCoordsFile = open(args.atlas,"rb")
		coords = []
		for line in mniCoordsFile.read().splitlines():
			splitedLine = line.decode().split(' ')
			newCoord = []
			for part in splitedLine:
				if part != '':
					newCoord.append(float(part))
			coords.append(newCoord)
		mniCoordsFile.close()
	
	if (args.networks != None):
		if(args.networks == ["all"]):
			plotAllCombinations(df_corr_mat, "correlation_matrix" , args.out_folder, coords, args.min_r)
		else:
			plotSomeNetworks(df_corr_mat, "correlation_matrix", args.out_folder,args.networks, coords, args.min_r)
			