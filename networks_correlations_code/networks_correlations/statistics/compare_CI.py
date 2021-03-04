import argparse
import numpy as np
import pandas as pd
import os
import networks_correlations.visualize_functions as vf
import networks_correlations.networkToIndexDic as net_dic

def is_overlapping(start1, end1, start2, end2):
    return (start1 <= start2 <= end1 or
        start1 <= end2 <= end1 or
        start2 <= start1 <= end2 or
        start2 <= end1 <= end2)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--mat1', required=True, type=str, help='path to npy file with CIs matrix')
	parser.add_argument('--mat2', required=True, type=str, help='path to npy file with CIs matrix')
	args = parser.parse_args()
	
	with open(args.mat1, 'rb') as mat1_f:
		mat1 = np.load(mat1_f,allow_pickle=True)
	with open(args.mat2, 'rb') as mat2_f:
		mat2 = np.load(mat2_f,allow_pickle=True)
		
	#Check dimensions
	if mat1.shape != mat2.shape:
		print ("Error in matrices dimensions!")
		exit(1)
		
	num_of_parcels = mat1.shape[0]
	different_ROIs = np.empty((num_of_parcels,num_of_parcels), dtype = bool)
	lower = 0
	upper = 1
	for i in range(num_of_parcels):
		for j in range(num_of_parcels):
			different_ROIs[i,j] = not is_overlapping(mat1[i,j][lower],mat1[i,j][upper],mat2[i,j][lower],mat2[i,j][upper])
		
	#pd.DataFrame(mat1).to_excel(os.path.join(args.out_folder,'mat1.xlsx'),index=False,header=False)
	#pd.DataFrame(mat2).to_excel(os.path.join(args.out_folder,'mat2.xlsx'),index=False,header=False)
	with open(os.path.join(args.out_folder,'diff_ROIs.npy'), 'wb') as f:
		np.save(f, different_ROIs)
	print(different_ROIs)
	