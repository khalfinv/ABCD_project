import os, sys
import networkToIndexDic 
import runCommonMat as cm
import pickle
from config import USE_PREV_RESULTS 
from nilearn import plotting
import matplotlib.pyplot as plt

#get input
numOfArgs = len(sys.argv[1:])
argList = sys.argv[1:]
print(numOfArgs)
print(argList)

if USE_PREV_RESULTS is False:
    (common_cov_mat, common_cor_mat) = cm.createCommonMat()
    print("Write to pkl file")
    f = open('commonCovAndCorMat.pkl', mode="wb")
    pickle.dump((common_cov_mat, common_cor_mat), f)
    f.close()  
else:
    print("Read from pkl file")
    pkl_file = open('commonCovAndCorMat.pkl', 'rb')
    (common_cov_mat, common_cor_mat) = pickle.load(pkl_file)
    pkl_file.close()
	
#extract coordinates from power atlas
mniCoordsFile = open("MNI_Power.txt","rb")
coords = []
for line in mniCoordsFile.read().splitlines():
    splitedLine = line.decode().split(' ')
    newCoord = []
    for part in splitedLine:
        if part is not '':
            newCoord.append(float(part))
    coords.append(newCoord)
mniCoordsFile.close()

labels = []
#In case of at least one network in arguments
if(numOfArgs > 0):
    #create a list of indecis to slice
    listToSlice = []
    for network in argList:
        print (network)
        if network in networkToIndexDic.dic:
            labels = labels + [network] * len(list(networkToIndexDic.dic[network]))
            listToSlice = listToSlice + list(networkToIndexDic.dic[network])
        else:
            print ( "The " + network + " network does not exist!!!")
    common_cov_mat = common_cov_mat[listToSlice, :][:, listToSlice] 
    common_cor_mat = common_cor_mat[listToSlice, :][:, listToSlice]
    coords = [coords[i] for i in listToSlice]	
else:
    print ("Insert at least one network")
    exit()

#create folder for the output if not exists 
common_mat_folder = "CommonMatrices"
if not os.path.exists(common_mat_folder):
    os.makedirs(common_mat_folder)
	
#Plot the common matrices
fig = plt.figure()
plotting.plot_matrix(common_cov_mat, colorbar=True, labels = labels, figure=fig, title='Common covariance matrix')
fig.savefig(common_mat_folder + "/common_cov_matrix_" + str(argList) + ".png")
fig2 = plt.figure()
plotting.plot_matrix(common_cor_mat, colorbar=True, labels= labels, vmin=-1., vmax=1., figure=fig2, title='Common correlation matrix')
fig2.savefig(common_mat_folder + "/common_cor_matrix_" + str(argList) + ".png")
fig3 = plt.figure()
plotting.plot_connectome(adjacency_matrix= common_cor_mat, node_coords= coords, edge_threshold="80%", colorbar=True,edge_vmin = -1, edge_vmax = 1, figure=fig3, title= str(argList))
fig3.savefig(common_mat_folder + "/brain_plot_" + str(argList) + ".png")
view = plotting.view_connectome(common_cor_mat, coords, threshold="80%")    
view.open_in_browser()