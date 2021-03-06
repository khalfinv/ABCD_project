import os, sys, argparse
import pandas as pd
import networks_correlations.visualize_functions as vf
import networks_correlations.networkToIndexDic as net_dic


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', required=True, type=str, help='output folder')
    parser.add_argument('--mat1', required=True, type=str, help='path to excel file containing the matrix 1')
    parser.add_argument('--mat2', required=True, type=str, help='path to excel file containing the matrix 2')
    parser.add_argument('--out_name', default="sig_values", required=False, type=str, help='Name for output files')
    #parser.add_argument('--mat3', required=True, type=str, help='path to excel file containing the matrix 3')
    args = parser.parse_args()

    df_mat1 = pd.read_excel(args.mat1,index_col= 0)
    df_mat2 = pd.read_excel(args.mat2,index_col= 0)
    #df_mat3 = pd.read_excel(args.mat3,index_col= 0)
    mat1 = df_mat1.notnull().to_numpy()
    mat2 = df_mat2.notnull().to_numpy()
    #mat3 = df_mat3.notnull().to_numpy()
    #mat1_mat2_mat3 = mat1*mat2*mat3
    mat1_mat2 = mat1*mat2
    print(mat1_mat2)
    network = "Default"
    ticks = [0]
    ticks.append(ticks[-1] + len(net_dic.dic[network]))
    vf.plotMatrix(mat1_mat2, os.path.join(args.out_folder,args.out_name + ".png"), [network], "Significant Values Two Matrices",ticks)

    pd.DataFrame(mat1_mat2).to_excel(os.path.join(args.out_folder,args.out_name + ".xlsx"))