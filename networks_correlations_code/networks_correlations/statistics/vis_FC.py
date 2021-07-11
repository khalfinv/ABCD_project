import pandas as pd
import numpy as np
import networks_correlations.visualize_functions as vf
import networks_correlations.networkToIndexDic as net_dic
import os, sys, pickle, argparse


def plot_FC(matrix, coords, title, out_folder, out_name):
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
	vf.plotMatrix(matrix.values, os.path.join(out_folder,out_name) , [network], "Matrix 1",ticks)
	#Plot brain connectome if all coordinates exists for those networks
	if(max(listToSlice) < len(coords)):
		coords_sliced = [coords[i] for i in listToSlice]
		vf.plotConnectome(matrix.values, coords_sliced, [network], out_folder, title, min_r)
		
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--sig_mat_pos', required=True, type=str, help='path to significangt ROIs matrix with boolean values - positive')
	parser.add_argument('--sig_mat_neg', required=True, type=str, help='path to significangt ROIs matrix with boolean values - negative')
	parser.add_argument('--mat1', required=True, type=str, help='path to excel file containing the correlation matrix 1')
	parser.add_argument('--mat2', required=True, type=str, help='path to excel file containing the correlation matrix 2')
	parser.add_argument('--atlas', required=False, help='path to atlas coordinates file')
	parser.add_argument('--out_folder', required=True, help='path to output folder')
	args = parser.parse_args()
	
	df_mat_pos = pd.read_excel(args.sig_mat_pos,index_col= 0)
	df_mat_neg = pd.read_excel(args.sig_mat_neg,index_col= 0)
	df_mat1 = pd.read_excel(args.mat1,index_col= 0)
	df_mat2 = pd.read_excel(args.mat2,index_col= 0)
	#Positive matrices
	mat1_pos = df_mat1.to_numpy()*df_mat_pos.to_numpy()
	mat2_pos = df_mat2.to_numpy()*df_mat_pos.to_numpy()
	
	#Negative matrices
	mat1_neg = df_mat1.to_numpy()*df_mat_neg.to_numpy()*-1
	mat2_neg = df_mat2.to_numpy()*df_mat_neg.to_numpy()*-1
	
	#Combine positive and negative matrix
	df_mat1 = pd.DataFrame(mat1_neg + mat1_pos).replace(0,np.nan)
	df_mat2 = pd.DataFrame(mat2_neg + mat2_pos).replace(0,np.nan)
	
	df_mat1.to_excel(args.out_folder + "\mat1.xlsx")
	df_mat2.to_excel(args.out_folder + "\mat2.xlsx")
	
	#Visualization- need to modify
	#extract coordinates from atlas
	mniCoordsFile = open(args.atlas,"rb")
	coords = []
	for line in mniCoordsFile.read().splitlines():
		splitedLine = line.decode().split()
		newCoord = []
		for part in splitedLine:
			if part != '':
				newCoord.append(float(part))
		coords.append(newCoord)
	mniCoordsFile.close()
	
	plot_FC(df_mat1,coords,"Matrix 1", args.out_folder, "mat1.png")
	plot_FC(df_mat2,coords,"Matrix 2", args.out_folder, "mat2.png")
	
	
	
	#Only red or blue, without range
	mat1_neg[mat1_neg != 0] = -0.6
	mat2_neg[mat2_neg != 0] = -0.6
	mat1_pos[mat1_pos != 0] = 0.6
	mat2_pos[mat2_pos != 0] = 0.6

	#Combine positive and negative matrix
	df_mat1 = pd.DataFrame(mat1_neg + mat1_pos).replace(0,np.nan)
	df_mat2 = pd.DataFrame(mat2_neg + mat2_pos).replace(0,np.nan)
	
	plot_FC(df_mat1,coords,"Matrix 1 no range", args.out_folder, "mat1_without_range.png")
	plot_FC(df_mat2,coords,"Matrix 2 no range", args.out_folder, "mat2_without_range.png")
	

	

	
	

		
		
	