import os, sys, argparse
import pandas as pd
import networks_correlations.visualize_functions as vf
import networks_correlations.networkToIndexDic as net_dic
import numpy as np
from functools import reduce

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', required=True, type=str, help='output folder')
    parser.add_argument('--mat', required=True, nargs='+', help='path to all matrices')
    parser.add_argument('--out_name', default="sig_values", required=False, type=str, help='Name for output files')
    args = parser.parse_args()

    all_mat = []
    for mat in args.mat:
        mat_n = pd.read_excel(mat,index_col= 0).to_numpy()
        all_mat.append(mat_n)
        
    
    all_mat_mul = reduce((lambda x, y: x * y), all_mat)
    print(all_mat_mul)
    # df_mat1 = pd.read_excel(args.mat1,index_col= 0)
    # df_mat2 = pd.read_excel(args.mat2,index_col= 0)
    # #df_mat3 = pd.read_excel(args.mat3,index_col= 0)
    # mat1 = df_mat1.notnull().to_numpy()
    # mat2 = df_mat2.notnull().to_numpy()
    # #mat3 = df_mat3.notnull().to_numpy()
    # #mat1_mat2_mat3 = mat1*mat2*mat3
    # mat1_mat2 = mat1*mat2
    # print(mat1_mat2)
    network = "Default"
    ticks = [0]
    ticks.append(ticks[-1] + len(net_dic.dic[network]))
    vf.plotMatrix(all_mat_mul, os.path.join(args.out_folder,args.out_name + ".png"), [network], "Significant Values Two Matrices",ticks)

    pd.DataFrame(all_mat_mul).to_excel(os.path.join(args.out_folder,args.out_name + ".xlsx"))