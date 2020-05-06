import networkToIndexDic 
from subprocess import call
import itertools
import os
import multiprocessing as mp
from nilearn import plotting
import matplotlib.pyplot as plt
import time
import runCommonMat as cm
import pickle
from config import USE_PREV_RESULTS 

start = time.time()

if USE_PREV_RESULTS is False:
    (common_cov_mat_orig, common_cor_mat_orig) = cm.createCommonMat()
    print("Write to pkl file")
    f = open('commonCovAndCorMat.pkl', mode="wb")
    pickle.dump((common_cov_mat_orig, common_cor_mat_orig), f)
    f.close()  
else:
    print("Read from pkl file")
    pkl_file = open('commonCovAndCorMat.pkl', 'rb')
    (common_cov_mat_orig, common_cor_mat_orig) = pickle.load(pkl_file)
    pkl_file.close()	

#create folder for the output if not exists 
common_mat_folder = "CommonMatrices"
if not os.path.exists(common_mat_folder):
    os.makedirs(common_mat_folder)
	
#extract coordinates from power atlas
mniCoordsFile = open("MNI_Power.txt","rb")
coords_orig = []
for line in mniCoordsFile.read().splitlines():
    splitedLine = line.decode().split(' ')
    newCoord = []
    for part in splitedLine:
        if part is not '':
            newCoord.append(float(part))
    coords_orig.append(newCoord)
mniCoordsFile.close()
	
for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
    labels = []
    #create a list of indexes to slice
    listToSlice = []
    network1 = (pair)[0]
    network2 = (pair)[1]
    print (network1, network2)
    labels = labels + [network1] * len(list(networkToIndexDic.dic[network1]))
    listToSlice = listToSlice + list(networkToIndexDic.dic[network1])
    labels = labels + [network2] * len(list(networkToIndexDic.dic[network2]))
    listToSlice = listToSlice + list(networkToIndexDic.dic[network2])
    common_cov_mat_sliced = common_cov_mat_orig[listToSlice, :][:, listToSlice] 
    common_cor_mat_sliced = common_cor_mat_orig[listToSlice, :][:, listToSlice] 
    coords_sliced = [coords_orig[i] for i in listToSlice]
	#Plot the common matrices
    fig = plt.figure()
    plotting.plot_matrix(common_cov_mat_sliced, colorbar=True, labels = labels, figure=fig, title='Common covariance matrix')
    fig.savefig(common_mat_folder + "/common_cov_matrix_" + network1 + "_" + network2 + ".png")
    fig2 = plt.figure()
    plotting.plot_matrix(common_cor_mat_sliced, colorbar=True, labels= labels, vmin=-1., vmax=1., figure=fig2, title='Common correlation matrix')
    fig2.savefig(common_mat_folder + "/common_cor_matrix_" + network1 + "_" + network2 + ".png")
    fig3 = plt.figure()
    plotting.plot_connectome(adjacency_matrix= common_cor_mat_sliced, node_coords= coords_sliced, edge_threshold="80%", colorbar=True,edge_vmin = -1, edge_vmax = 1, figure=fig3, title= network1 + " and " + network2)
    fig3.savefig(common_mat_folder + "/brain_plot_" + network1 + "_" + network2 + ".png")
    #view = plotting.view_connectome(common_cor_mat_sliced, coords_sliced, threshold="80%")    
    #view.open_in_browser() 
    plt.close()

	
#print total time
end = time.time()
timeInSeconds = (end - start)
timeInMinutes = timeInSeconds / 60
timeInHours = int(timeInMinutes / 60)

print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")