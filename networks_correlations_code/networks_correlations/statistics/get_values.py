import pandas as pd
import numpy as np
import networks_correlations.visualize_functions as vf
import networks_correlations.networkToIndexDic as net_dic

if __name__ == "__main__":
	df_mat_bool = pd.read_excel(r"C:\Users\ויקי\Desktop\ABCD_project\Class4-Class2_sig\sig_values_all_mat.xlsx",index_col= 0)
	df_mat1 = pd.read_excel(r"C:\Users\ויקי\Desktop\ABCD_project\group2\Class4-Class2\sig_values_mat_1.xlsx",index_col= 0)
	df_mat2 = pd.read_excel(r"C:\Users\ויקי\Desktop\ABCD_project\group2\Class4-Class2\sig_values_mat_2.xlsx",index_col= 0)
	out = r"C:\Users\ויקי\Desktop\ABCD_project\Class4-Class2_sig\group2"
	df_mat1 = df_mat1.to_numpy()*df_mat_bool.to_numpy()
	df_mat1 = pd.DataFrame(df_mat1).replace(0,np.nan)
	df_mat2 = df_mat2.to_numpy()*df_mat_bool.to_numpy()
	df_mat2 = pd.DataFrame(df_mat2).replace(0,np.nan)
	df_mat1.to_excel(out + "\mat1.xlsx")
	df_mat2.to_excel(out + "\mat2.xlsx")
	
	#Visualization- need to modify
	#extract coordinates from atlas
	mniCoordsFile = open("../../Atlases/MNI_Gordon.txt","rb")
	coords = []
	for line in mniCoordsFile.read().splitlines():
		splitedLine = line.decode().split()
		newCoord = []
		for part in splitedLine:
			if part != '':
				newCoord.append(float(part))
		coords.append(newCoord)
	mniCoordsFile.close()
	
	
	#Slice DMN
	ticks = [0]
	min_r = 0
	#create a list of indexes to slice
	listToSlice = []
	network = "Default"
	# The networkToIndexDic dictionary contains for each network the location (indexes) in the common matrix.
	listToSlice = listToSlice + list(net_dic.dic[network])
	ticks.append(ticks[-1] + len(net_dic.dic[network]))
	#plot- only significant
	vf.plotMatrix(df_mat1.values, out+ "/mat_1.png", [network], "Matrix 1",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(df_mat1.values, coords_sliced, [network], out, "matrix1", min_r)
		
	#plot- only significant
	vf.plotMatrix(df_mat2.values, out+ "/mat_2.png", [network], "Matrix 2",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(df_mat2.values, coords_sliced, [network], out, "matrix2", min_r)
		
		
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(df_mat_bool.values, coords_sliced, [network], out, "sig_fc", min_r)
	