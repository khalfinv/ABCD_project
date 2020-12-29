#!/usr/bin/python3
"""
==============================================================================================================
Visualize functions for matrix and brain plotting 
==============================================================================================================

"""
from nilearn import plotting
import os, sys, pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import networkToIndexDic
import itertools

def plotConnectome(matrix, coords, networks, out_folder, base_name, min_r):
	"""Creates brain plot with connections according to minimum R value
	param matrix: two dimensional array. The correlation matrix
	param coords : list. Coordinates of the networks
	param networks: list. Networks names.
	param out_folder: string. Output folder for brain plot
	param base_name: string. Base name
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
	plot_name = out_folder + "/" + base_name + "_brain_plot" + str(networks) + ".png"
	if(len(correlated_coords) > 0):
		fig = plt.figure()
		title = "Threshold : " + str(min_r)
		plotting.plot_connectome(adjacency_matrix= matrix, node_coords= coords, title = title, node_color = colors, colorbar=True, edge_threshold=min_r ,
			node_size = 5, edge_vmin = -1, edge_vmax = 1, figure=fig, display_mode  = 'lyrz')
		fig.legend(handles=patches_list, loc = "upper center")
		fig.savefig(plot_name)
		plt.close()
	view = plotting.view_connectome(matrix, coords, edge_threshold=min_r, node_size = 7) 
	html_name = out_folder + "/" + base_name + "_brain_plot" + str(networks) + ".html"
	view.save_as_html(html_name)
	
	
def plotMatrix(matrix, plot_path, labels, title, ticks, vmin = -1., vmax = 1.):
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
		
	
	
if __name__ == "__main__":
	print("Visualization functions")
			