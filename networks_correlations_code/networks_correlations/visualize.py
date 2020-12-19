
from nilearn import plotting
import os, sys, pickle, argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import networkToIndexDic

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
	view = plotting.view_connectome(matrix, coords, edge_threshold=min_r, node_size = 7) 
	html_name = out_folder + "/brain_plot" + str(networks) + ".html"
	view.save_as_html(html_name)
	
	
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
	ax.xaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
	ax.yaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
	ax.grid(color='black', linestyle='-', linewidth=1.2, which='minor')
	plt.title(label = title, fontsize = 20)
	plotting.plot_matrix(matrix, colorbar=True, axes = ax, vmin=vmin, vmax=vmax)
	plt.yticks(ticks_middle,list(labels))
	plt.xticks(ticks_middle,list(labels), rotation = 55, horizontalalignment='right')
	for item in (ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_color(networkToIndexDic.labelToColorDic[item.get_text()])
		item.set_fontsize(14)
	fig.savefig(plot_path)
	plt.close()
	
def plotAllCombinations(common_cor_mat, out_folder, coords, min_r):
	"""Create plots of ovariance and correlation matrices for all combinations of 2 networks. 
	param common_cor_mat: two dimensional array (264, 264).The common correlation matrix with all networks
	param out_folder: string. The path of output folder
	param coords: list.List of atlas coordinates
	return: None
	"""
	print("Plot correlations between every two networks")
	for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
		#create a list of indexes to slice
		listToSlice = []
		network1 = (pair)[0]
		network2 = (pair)[1]
		#print (network1, network2)
		# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix. 		
		listToSlice = listToSlice + list(networkToIndexDic.dic[network1])
		listToSlice = listToSlice + list(networkToIndexDic.dic[network2])
		common_cor_mat_sliced = common_cor_mat[listToSlice, :][:, listToSlice]		
		ticks = [0, len(networkToIndexDic.dic[network1]), len(networkToIndexDic.dic[network1]) + len(networkToIndexDic.dic[network2])]                                                                                                                                                                                                                                              
		plotMatrix(common_cor_mat_sliced, out_folder + "/common_cor_matrix_" + network1 + "_" + network2 + ".png", [network1, network2], "Common correlation matrix - " + network1 + "_" + network2, ticks, -1., 1.)
		#Plot brain connectome if all coordinates exists for those networks
		if(max(listToSlice) < len(coords)):
			coords_sliced = [coords[i] for i in listToSlice]
			plotConnectome(common_cor_mat_sliced, coords_sliced, pair, out_folder, min_r)

				   
def plotSomeNetworks(common_cor_mat,out_folder, networks , coords, min_r):
	"""Create plots of covariance and correlation matrices for specific networks 
	param common_cor_mat : two dimensional array (number of parcels, number of parcels).The common correlation matrix with all networks
	param out_folder: string. The path of output folder
	param networks: list. List of networks for matrices generating
	return: None
	"""
	print("Plot correlation for specified networks")
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
	plotMatrix(common_cor_mat_sliced, out_folder + "/common_cor_matrix_" + str(networks) + ".png", networks, "Common correlation matrix - " + str(networks), ticks, -1., 1.)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		plotConnectome(common_cor_mat_sliced, coords_sliced, networks, out_folder, min_r)
		
def plotAllNetworks(common_cor_mat,out_folder):
	print("Plot correlation common matrix")
	ticks = [0]
	#create a list of indexes by networks order
	listOfIndexes = []
	for network in networkToIndexDic.dic.keys():
		#print (network)
		if network in networkToIndexDic.dic:
			# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
			listOfIndexes = listOfIndexes + list(networkToIndexDic.dic[network])
			ticks.append(ticks[-1] + len(networkToIndexDic.dic[network]))
		else:
			print ( "The " + network + " network does not exist!!!")
	common_cor_mat_new = common_cor_mat[listOfIndexes, :][:, listOfIndexes] 
	plotMatrix(common_cor_mat_new, out_folder + "/common_cor_matrix.png", networkToIndexDic.dic.keys(), "Common correlation matrix", ticks, -1., 1.)
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder for common matrices pkl file (if generated) and plots')
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
	common_cor_mat = df_corr_mat.values
	plotAllNetworks(common_cor_mat,args.out_folder)
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
			plotAllCombinations(common_cor_mat, args.out_folder, coords, args.min_r) 
		else:
			plotSomeNetworks(common_cor_mat,args.out_folder,args.networks, coords, args.min_r)