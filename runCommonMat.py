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

@Output:
commonCovAndCorMat.pkl file: located in out_folder. This file contains covariance and correlation common matrices of size (264, 264): (common_cov_mat, common_cor_mat).
							Will be genereated if use_prev is False (does not exist in input)
common_cov_matrix.png file: located in out_folder. This file contains the plot of the common covariance matrix. 
common_cor_matrix.png file: located in out_folder. This file contains the plot of the common correlation matrix. 			

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
		
	
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', required=True, type=str, help='output folder for common matrices pkl file (if generated) and plots')
    parser.add_argument('--exclusion_criteria', required=False, type=int, default=375, help='Number of volumes for subjects exclusion. Default is 375 (5 minutes scan)')
    parser.add_argument('--use_prev', help='use previous generated covariance and correlation matrices',action='store_true')
    parser.add_argument('--common_mat_pkl', required=False, type=str, help='path to common matrices pkl file, mandatory only if use_prev flag exists')
    parser.add_argument('--subjects_data_dict', required=False, type=str, help='path to pkl file containing the subjects data dictionary, mandatory only if use_prev flag does not exist')
    args = parser.parse_args()
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
	
	#Plot the common matrices
    print("Plot the common matrices")
    ticks = [0,28,58,63,77,90,148,153,184,209,227,240,249,260,264]
    ticks_middle = [(((ticks[i+1]-ticks[i]) / 2 ) + ticks[i]) for i in range(0,len(ticks)-1)]
    labelToColorDic = {"Uncertain" : "pink", "Somatomotor Hand" : "cyan", "Somatomotor Mouth" : "orange", "Cingulo-opercular" : "purple", "Auditory" : "m", "Default mode" : "red", "Memory" : "grey", 
	"Visual" : "blue", "Fronto-parietal" : "gold", "Salience" : "black", "Subcortical" : "brown", "Ventral attention" : "teal", "Dorsal attention" : "green", "Cerebellum" : "purple"}
    fig, ax = plt.subplots()
    fig.set_size_inches(16.5, 9.5)
    plt.yticks(ticks_middle,labelToColorDic.keys())
    plt.xticks(ticks_middle,labelToColorDic.keys(), rotation = 55, horizontalalignment='right')
    ax.xaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
    ax.yaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
    ax.grid(color='black', linestyle='-', linewidth=1.2, which='minor')
    plt.title(label = 'Common covariance matrix', fontsize = 20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_color(labelToColorDic[item.get_text()])
        item.set_fontsize(12)
    plotting.plot_matrix(common_cov_mat, colorbar=True, figure=fig)
    fig.savefig(args.out_folder + "/common_cov_matrix.png")
	
    fig2, ax = plt.subplots()
    fig2.set_size_inches(16.5, 9.5)
    plt.yticks(ticks_middle,labelToColorDic.keys())
    plt.xticks(ticks_middle,labelToColorDic.keys(), rotation = 55, horizontalalignment='right')
    ax.xaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
    ax.yaxis.set_minor_locator(plt.FixedLocator(ticks[1:]))
    ax.grid(color='black', linestyle='-', linewidth=1.2, which='minor')
    plt.title(label = 'Common correlation matrix', fontsize = 20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_color(labelToColorDic[item.get_text()])
        item.set_fontsize(12)
    plotting.plot_matrix(common_cor_mat, colorbar=True, vmin=-1., vmax=1., figure=fig2)
    fig2.savefig(args.out_folder + "/common_cor_matrix.png")
 
	

