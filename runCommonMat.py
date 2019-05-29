import os, sys, pickle
from common_statistics import snr, est_common_cov, est_common_density2D
from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np
import nilearn
from nilearn.connectome import cov_to_corr
from config import USE_PREV_RESULTS 
import matplotlib.gridspec as gridspec


def createCommonMat():
    #read python dict back from the file
    pkl_file = open('covMatAndTimeSerias.pkl', 'rb')
    dict = pickle.load(pkl_file)
    pkl_file.close()
    #covariance matrices
    covars = [val["covariance"] for val in dict.values()] 
    #find common covariance matrix
    common_cov_mat = est_common_cov(covars)
    #create common correlation matrix
    common_cor_mat = nilearn.connectome.cov_to_corr(common_cov_mat)
    return (common_cov_mat, common_cor_mat)
		
	
if __name__ == "__main__":
    if USE_PREV_RESULTS is False:
        (common_cov_mat, common_cor_mat) = createCommonMat()
        print("Write to pkl file")
        f = open('commonCovAndCorMat.pkl', mode="wb")
        pickle.dump((common_cov_mat, common_cor_mat), f)
        f.close() 
    else:
        print("Read from pkl file")
        pkl_file = open('commonCovAndCorMat.pkl', 'rb')
        (common_cov_mat, common_cor_mat) = pickle.load(pkl_file)
        pkl_file.close()
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
    fig.savefig("common_cov_matrix.png")
	
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
    fig2.savefig("common_cor_matrix.png")
 
	

