#!/usr/bin/python3
"""
==============================================================================================================
Create common covariance and correlation matrix according to Zhitnikov et al. and plot the correlation matrix.
==============================================================================================================

@Input:  
out_folder = string. Output folder for excel files and plots
subjects_data_dict = string. Path to pkl file containing the subjects' data dictionary
networks = list, not mandatory. Generate covarinace and correlation matrices for the networks specified. 
           To create all the combinations of two networks use --networks all. 
		   The networks' names are according to networkToIndexDic dictionary.

@Output: 
common_cor_matrix.png: located in out_folder. This file contains the plot of the common correlation matrix. 
correlation_matrix.xlsx: located in out_folder. Excel file with the common correlation matrix.
correlation_sum.xlsx: located in out_folder. Excel file with correlation score for between and within networks.
brain plots and correlation matrix plots for the networks specified in --networks flag	
		

"""

import os, sys, pickle, argparse
from common_statistics import snr, est_common_cov, est_common_density2D
from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np
import nilearn
from nilearn.connectome import cov_to_corr 
import matplotlib.gridspec as gridspec
import time
import networkToIndexDic
import itertools
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd



def createCommonMat(subjects_data_dict):
	"""Create common covariance and correlation matrices
	param subjects_data_dict: dictionary. All subject's data dictionary from post processing step
	return: (common_cov_mat, common_cor_mat) : tuple. 
		common_cov_mat - common covariance matrix (264, 264), common_cor_mat - common correlation matrix (264, 264)
	"""
	print("Create common correlation matrix")
	#covariance matrices
	covars = []
	for val in subjects_data_dict.values():
		covars.append(val["covariance"])
	print("number of subjects: ", len(covars))
	#find common covariance matrix
	common_cov_mat = est_common_cov(covars)
	#create common correlation matrix
	common_cor_mat = nilearn.connectome.cov_to_corr(common_cov_mat)
	return (common_cov_mat, common_cor_mat)

def plotMatrix(matrix, plot_path, labels, title, ticks, vmin, vmax):
	"""Plot matrix. 
	param matrix: two dimensional array. The matrix to plot
	param plot_path : string. Full path and name of the plotting picture
	param labels: list. The labels 
	param title: string. The title of the plot
	param ticks: list. The ticks of the plot
	vmin: float. Minimum value
	vmax: float. Maximum value
	return: None
	"""
	ticks = list(map(lambda x: x - 0.5, ticks))
	ticks_middle = [(((ticks[i+1]-ticks[i]) / 2 ) + ticks[i]) for i in range(0,len(ticks)-1)]
	fig, ax = plt.subplots()
	fig.set_size_inches(16.5, 9.5)
	plt.yticks(ticks_middle,list(labels))
	plt.xticks(ticks_middle,list(labels), rotation = 55, horizontalalignment='right')
	ax.xaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
	ax.yaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
	ax.grid(color='black', linestyle='-', linewidth=1.2, which='minor')
	plt.title(label = title, fontsize = 20)
	for item in (ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_color(networkToIndexDic.labelToColorDic[item.get_text()])
		item.set_fontsize(14)
	plotting.plot_matrix(matrix, colorbar=True, axes = ax, vmin=vmin, vmax=vmax)
	fig.savefig(plot_path)
	plt.close()


def sumCorrScore(out_folder, common_cor_mat):
	"""Sun correlation scores for within and between networks and save the results to excel file. 
	param out_folder: string. Output folder for excel file
	param common_cor_mat : two dimensional array. The correlation matrix
	return: None
	"""
	df_raws = []
	#Calculate between correlations
	print("Calculating between correlations")
	for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
		new_raw = {}
		#create a list of indexes to slice
		listToSlice = []
		network1 = (pair)[0]
		network2 = (pair)[1]
		r_sum = 0
		count = 0
		for i in networkToIndexDic.dic[network1]:
			for j in networkToIndexDic.dic[network2]:
				r_sum = r_sum + common_cor_mat[i][j]
				count = count + 1
		mean_corr = r_sum / count
		new_raw['networks name'] = network1 + '_' + network2
		new_raw['correlation score'] = mean_corr
		df_raws.append(new_raw)
		#print ("r_sum: ", r_sum, " count: ", count, " mean: ", mean_corr)
		
	#Calculate within correlations 
	print("Calculating within correlations")
	for network in networkToIndexDic.dic.keys():
		new_raw = {}
		r_sum = 0
		count = 0
		start_index = networkToIndexDic.dic[network][0]
		end_index = networkToIndexDic.dic[network][-1] + 1
		for i in range(start_index + 1, end_index) :
			for j in range(start_index, i):
				r_sum = r_sum + common_cor_mat[i][j]
				count = count + 1
		mean_corr = r_sum / count
		new_raw['networks name'] = network + '_' + network
		new_raw['correlation score'] = mean_corr
		df_raws.append(new_raw)
		#print ("r_sum: ", r_sum, " count: ", count, " mean: ", mean_corr)
		
	#Save to excel file
	df = pd.DataFrame(df_raws) 
	df.to_excel(out_folder + "/correlation_sum.xlsx", index=False)

	
def plotConnectome(matrix, coords, networks, out_folder, min_r):
	"""Creates brain plot with connections according to minimum R value
	param matrix: two dimensional array. The correlation matrix
	param coords : list. Coordinates of thye networks
	param networks: list. Networks names.
	param out_folder: string. Output folder for brain plot
	param min_r: float. The minimun R value for connection plotting.
	return: None
	"""
	colors = []
	patches_list = []
	index2network = {}
	last_index = 0
	for network in networks:
		network_color = [networkToIndexDic.labelToColorDic[network]] * len(list(networkToIndexDic.dic[network]))
		colors = colors + network_color
		for i in range(last_index, last_index + len(list(networkToIndexDic.dic[network]))):
			index2network[i] = network
			last_index = i+1
		network_patch = mpatches.Patch(color=networkToIndexDic.labelToColorDic[network], label=network)
		patches_list.append(network_patch)
	correlated_coords = {} 
	for i in range(1,len(matrix)):
		for j in range(i):
			if(matrix[i][j] >= min_r):
				correlated_coords[matrix[i][j]] = ({index2network[i] : coords[i]}, {index2network[j] : coords[j]})
	plot_name = out_folder + "/brain_plot" + str(networks) + ".png"
	if(len(correlated_coords) > 0):
		fig = plt.figure()
		title = "Threshold : " + str(min_r)
		plotting.plot_connectome(adjacency_matrix= matrix, node_coords= coords, title = title, node_color = colors, colorbar=True, edge_threshold=min_r ,
			node_size = 5, edge_vmin = -1, edge_vmax = 1, figure=fig, display_mode  = 'lyrz')
		fig.legend(handles=patches_list, loc = "upper center")
		fig.savefig(plot_name)
		plt.close()
	
def runAllCombinations(common_cov_mat, common_cor_mat, out_folder, coords, min_r):
	"""Create plots of ovariance and correlation matrices for all combinations of 2 networks. 
	param common_cov_mat: two dimensional array (264,264). The common covariance matrix with all networks
	param common_cor_mat: two dimensional array (264, 264).The common correlation matrix with all networks
	param out_folder: string. The path of output folder
	param coords: list.List of atlas coordinates
	return: None
	"""
	print("Run correlations between every two networks")
	for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
		#create a list of indexes to slice
		listToSlice = []
		network1 = (pair)[0]
		network2 = (pair)[1]
		#print (network1, network2)
		# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix. 		
		listToSlice = listToSlice + list(networkToIndexDic.dic[network1])
		listToSlice = listToSlice + list(networkToIndexDic.dic[network2])
		common_cov_mat_sliced = common_cov_mat[listToSlice, :][:, listToSlice] 
		common_cor_mat_sliced = common_cor_mat[listToSlice, :][:, listToSlice]		
		coords_sliced = [coords[i] for i in listToSlice]
		ticks = [0, len(networkToIndexDic.dic[network1]), len(networkToIndexDic.dic[network1]) + len(networkToIndexDic.dic[network2])]                                                                                                                                                                                                                                              
		plotMatrix(common_cor_mat_sliced, out_folder + "/common_cor_matrix_" + network1 + "_" + network2 + ".png", [network1, network2], "Common correlation matrix", ticks, -1., 1.)
		plotConnectome(common_cor_mat_sliced, coords_sliced, pair, out_folder, min_r)

				   
def runSomeNetworks(common_cov_mat,common_cor_mat,out_folder, networks , coords, min_r):
	"""Create plots of covariance and correlation matrices for specific networks 
	param common_cov_mat: two dimensional array (264,264). The common covariance matrix with all networks
	param common_cor_mat : two dimensional array (264, 264).The common correlation matrix with all networks
	param out_folder: string. The path of output folder
	param networks: list. List of networks for matrices generating
	return: None
	"""
	print("Run correlation for specified networks")
	ticks = [0]
	#create a list of indexes to slice
	listToSlice = []
	for network in networks:
		#print (network)
		if network in networkToIndexDic.dic:
			# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
			listToSlice = listToSlice + list(networkToIndexDic.dic[network])
			ticks.append(ticks[-1] + len(networkToIndexDic.dic[network]))
		else:
			print ( "The " + network + " network does not exist!!!")
	common_cor_mat_sliced = common_cor_mat[listToSlice, :][:, listToSlice] 
	#coords_sliced = [coords[i] for i in listToSlice]
	plotMatrix(common_cor_mat_sliced, out_folder + "/common_cor_matrix_" + str(networks) + ".png", networks, "Common correlation matrix", ticks, -1., 1.)
	#plotConnectome(common_cor_mat_sliced, coords_sliced, networks, out_folder, min_r)

if __name__ == "__main__":
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder for common matrices pkl file (if generated) and plots')
	parser.add_argument('--subjects_data_dict', required=True, type=str, help='path to pkl file containing the subjects data dictionary')
	parser.add_argument('--networks', required=False, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
	parser.add_argument('--atlas', required=False, help='path to atlas coordinates file. Needed only if --networks flag exists')
	parser.add_argument('--min_r', required=False, type=float, default = 0.7, help='Minumun R value for brain connection plotting')
	args = parser.parse_args()
	
	#Arguments checks
	if (args.networks != None and args.atlas is None):
		print ("--atlas is required. Use -h for more details")
		exit(1)
	# generated new common covariance and correlation matrices
	pkl_file = open(args.subjects_data_dict, 'rb')
	allParticipantsDict = pickle.load(pkl_file)
	pkl_file.close()
	(common_cov_mat, common_cor_mat) = createCommonMat(allParticipantsDict)
	#Save common correlation to excel
	columns = []
	for network in networkToIndexDic.dic.keys():
		columns = columns + ([network] * len(list(networkToIndexDic.dic[network])))
	df = pd.DataFrame(common_cor_mat, columns = columns,  index=columns)
	df.to_excel(args.out_folder + "/correlation_matrix.xlsx")

	if args.atlas != None:
		#extract coordinates from power atlas
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
			runAllCombinations(common_cov_mat, common_cor_mat, args.out_folder, coords, args.min_r) 
		else:
			runSomeNetworks(common_cov_mat, common_cor_mat,args.out_folder,args.networks, coords, args.min_r)
	sumCorrScore(args.out_folder, common_cor_mat)
	#Plot the full common matrices
	#ticks = [0,28,58,63,77,90,148,153,184,209,227,240,249,260,264]
	ticks = [net[0] for net in networkToIndexDic.dic.values()]
	ticks.append(264)
	plotMatrix(common_cor_mat, args.out_folder + "/common_cor_matrix.png",networkToIndexDic.dic.keys(), "Common correlation matrix", ticks, -1., 1.)
	
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	