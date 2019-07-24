#!/usr/bin/python3
"""
==========================================================================================
Create common covariance and correlation matrix according to Zhitnikov et al. and plot them.
Use only the subjects that have number of volumes >= exclusion_criteria
==========================================================================================

@Input:  
out_folder = string. Output folder for common matrices pkl file (if generated) and plots
exclusion_criteria = integer. Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)
use_prev = boolean. Use previous generated covariance and correlation matrices
common_mat_pkl = string. Path to common matrices pkl file, mandatory only if use_prev flag exists
subjects_data_dict = string. Path to pkl file containing the subjects' data dictionary, mandatory only if use_prev flag does not exist
networks = list, not mandatory. Generate covarinace and correlation matrices for the networks specified. 
           To create all the combinations of two networks use --networks all. The networks' names are according to networkToIndexDic dictionary.

@Output:
commonCovAndCorMat.pkl file: located in out_folder. This file contains covariance and correlation common matrices of size (264, 264): (common_cov_mat, common_cor_mat).
							Will be genereated if use_prev is False (does not exist in input)
common_cov_matrix.png file: located in out_folder. This file contains the plot of the common covariance matrix. 
common_cor_matrix.png file: located in out_folder. This file contains the plot of the common correlation matrix. 
covariance and correlation networks specific png files according to --networks flag			

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


def createCommonMat(subjects_data_dict, exclusion_criteria):
    """Create common covariance and correlation matrices
	param subjects_data_dict: dictionary. All subject's data dictionary from post processing step
	param exclusion_criteria : integer. Exclude subjects that number of volumes are less than exclusion_criteria
	return: (common_cov_mat, common_cor_mat) : tuple. 
		common_cov_mat - common covariance matrix (264, 264), common_cor_mat - common correlation matrix (264, 264)
    """
    start = time.time()	
    #covariance matrices
    covars = []
    for val in subjects_data_dict.values():
        if val["num_of_volumes"] >= exclusion_criteria:
            covars.append(val["covariance"])
    print("number of subjects: ", len(covars))
    #find common covariance matrix
    common_cov_mat = est_common_cov(covars)
    #create common correlation matrix
    common_cor_mat = nilearn.connectome.cov_to_corr(common_cov_mat)
    #print the exucation time
    end = time.time()
    timeInSeconds = (end - start)
    timeInMinutes = timeInSeconds / 60
    timeInHours = int(timeInMinutes / 60)
    print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	
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
    labelToColorDic = {"Uncertain" : "olive", "SSH" : "cyan", "SSM" : "orange", "CO" : "purple", "Auditory" : "m", "DMN" : "red", "Memory" : "grey", 
	"Visual" : "blue", "FP" : "gold", "Salience" : "black", "Subcortical" : "brown", "VAN" : "teal", "DAN" : "green", "Cerebellum" : "purple"}
    ticks_middle = [(((ticks[i+1]-ticks[i]) / 2 ) + ticks[i]) for i in range(0,len(ticks)-1)]
    fig, ax = plt.subplots()
    fig.set_size_inches(16.5, 9.5)
    plt.yticks(ticks_middle,labels)
    plt.xticks(ticks_middle,labels, rotation = 55, horizontalalignment='right')
    ax.xaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
    ax.yaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
    ax.grid(color='black', linestyle='-', linewidth=1.2, which='minor')
    plt.title(label = title, fontsize = 20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_color(labelToColorDic[item.get_text()])
        item.set_fontsize(14)
    plotting.plot_matrix(matrix, colorbar=True, figure=fig, vmin=vmin, vmax=vmax)
    fig.savefig(plot_path)
    plt.close()
	
def runAllCombinations(common_cov_mat, common_cor_mat, out_folder):
    """Create plots of ovariance and correlation matrices for all combinations of 2 networks. 
	param common_cov_mat: two dimensional array (264,264). The common covariance matrix with all networks
	param common_cor_mat : two dimensional array (264, 264).The common correlation matrix with all networks
    param out_folder: string. The path of output folder
	return: None
    """
    for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
        #create a list of indexes to slice
        listToSlice = []
        network1 = (pair)[0]
        network2 = (pair)[1]
        print (network1, network2)
		# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix. 		
        listToSlice = listToSlice + list(networkToIndexDic.dic[network1])
        listToSlice = listToSlice + list(networkToIndexDic.dic[network2])
        common_cov_mat_sliced = common_cov_mat[listToSlice, :][:, listToSlice] 
        common_cor_mat_sliced = common_cor_mat[listToSlice, :][:, listToSlice] 
        ticks = [0, len(networkToIndexDic.dic[network1]), len(networkToIndexDic.dic[network1]) + len(networkToIndexDic.dic[network2])]                                                                                                                                                                                                                                              
        plotMatrix(common_cor_mat_sliced, out_folder + "/common_cor_matrix_" + network1 + "_" + network2 + ".png", [network1, network2], "Common correlation matrix", ticks, -1., 1.)
        plotMatrix(common_cov_mat_sliced, out_folder + "/common_cov_matrix_" + network1 + "_" + network2 + ".png", [network1, network2], "Common covariance matrix",
                   ticks, common_cov_mat_sliced.min(), common_cov_mat_sliced.max())
				   
def runSomeNetworks(common_cov_mat,common_cor_mat,out_folder, networks ):
    """Create plots of covariance and correlation matrices for specific networks 
	param common_cov_mat: two dimensional array (264,264). The common covariance matrix with all networks
	param common_cor_mat : two dimensional array (264, 264).The common correlation matrix with all networks
    param out_folder: string. The path of output folder
	param networks: list. List of networks for matrices generating
	return: None
    """
    ticks = [0]
    #create a list of indexes to slice
    listToSlice = []
    for network in networks:
        print (network)
        if network in networkToIndexDic.dic:
		    # The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
            listToSlice = listToSlice + list(networkToIndexDic.dic[network])
            ticks.append(ticks[-1] + len(networkToIndexDic.dic[network]))
        else:
            print ( "The " + network + " network does not exist!!!")
    common_cov_mat_sliced = common_cov_mat[listToSlice, :][:, listToSlice] 
    common_cor_mat_sliced = common_cor_mat[listToSlice, :][:, listToSlice] 
    plotMatrix(common_cor_mat_sliced, out_folder + "/common_cor_matrix_" + str(networks) + ".png", networks, "Common correlation matrix", ticks, -1., 1.)
    plotMatrix(common_cov_mat_sliced, out_folder + "/common_cov_matrix_" + str(networks) + ".png", networks, "Common covariance matrix", ticks,
             	common_cov_mat_sliced.min(), common_cov_mat_sliced.max())
				   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', required=True, type=str, help='output folder for common matrices pkl file (if generated) and plots')
    parser.add_argument('--exclusion_criteria', required=False, type=int, default=375, help='Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)')
    parser.add_argument('--use_prev', help='use previous generated covariance and correlation matrices',action='store_true')
    parser.add_argument('--common_mat_pkl', required=False, type=str, help='path to common matrices pkl file, mandatory only if use_prev flag exists')
    parser.add_argument('--subjects_data_dict', required=False, type=str, help='path to pkl file containing the subjects data dictionary, mandatory only if use_prev flag does not exist')
    parser.add_argument('--networks', required=False, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
    args = parser.parse_args()
	#Arguments checks
    if (args.use_prev == True and args.common_mat_pkl is None):
        print ("--common_mat_pkl is required. Use -h for more details")
        exit(1)
    elif (args.use_prev == False and args.subjects_data_dict is None):
        print ("--subjects_data_dict is required. Use -h for more details")
        exit(1)
	
    if args.use_prev is False:
	    # generated new common covariance and correlation matrices
        print("Read from subjects' data pkl")
        pkl_file = open(args.subjects_data_dict, 'rb')
        allParticipantsDict = pickle.load(pkl_file)
        pkl_file.close()
        (common_cov_mat, common_cor_mat) = createCommonMat(allParticipantsDict, args.exclusion_criteria)
        print("Write common matrices to pkl file")
        f = open(args.out_folder + '/commonCovAndCorMat.pkl', mode="wb")
        pickle.dump((common_cov_mat, common_cor_mat), f)
        f.close() 
    else:
        #use previous version of common matrices
        print("Read from common matrices pkl file")
        pkl_file = open(args.common_mat_pkl, 'rb')
        (common_cov_mat, common_cor_mat) = pickle.load(pkl_file)
        pkl_file.close()
	
    if (args.networks != None):
        if(args.networks == ["all"]):
            runAllCombinations(common_cov_mat, common_cor_mat, args.out_folder) 
        else:
            runSomeNetworks(common_cov_mat, common_cor_mat,args.out_folder,args.networks)

	#Plot the full common matrices
    ticks = [0,28,58,63,77,90,148,153,184,209,227,240,249,260,264]
    plotMatrix(common_cor_mat, args.out_folder + "/common_cor_matrix.png",networkToIndexDic.dic.keys(), "Common correlation matrix", ticks, -1., 1.)
    plotMatrix(common_cov_mat, args.out_folder + "/common_cov_matrix.png",networkToIndexDic.dic.keys(), "Common covariance matrix",
                   ticks, common_cov_mat.min(), common_cov_mat.max())