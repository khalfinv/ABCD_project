import os, sys, argparse, pickle
import pandas as pd
import subprocess
import time

def checkSubprocessRet(return_code):
	if (return_code != 0):
		print("Command failed!")
		exit(1)
	else:
		print("Command succeeded!")


if __name__ == "__main__":
    compare_mat_commands = ["networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class1-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class1-Class2\sig_values_mat_1.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class1-Class2\sig_values_mat_1.xlsx"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class1-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class1-Class2\sig_mat_1_0.3.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class1-Class2\sig_mat_1_0.3.xlsx --out_name sig_val_0.3"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class1-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class1-Class2\sig_mat_1_0.2.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class1-Class2\sig_mat_1_0.2.xlsx --out_name sig_val_0.2"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class1-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class1-Class2\sig_mat_1_0.1.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class1-Class2\sig_mat_1_0.1.xlsx --out_name sig_val_0.1"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class3-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class3-Class2\sig_values_mat_1.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class3-Class2\sig_values_mat_1.xlsx"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class3-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class3-Class2\sig_mat_1_0.3.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class3-Class2\sig_mat_1_0.3.xlsx --out_name sig_val_0.3"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class3-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class3-Class2\sig_mat_1_0.2.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class3-Class2\sig_mat_1_0.2.xlsx --out_name sig_val_0.2"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class3-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class3-Class2\sig_mat_1_0.1.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class3-Class2\sig_mat_1_0.1.xlsx --out_name sig_val_0.1"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class4-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class4-Class2\sig_values_mat_1.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class4-Class2\sig_values_mat_1.xlsx"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class4-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class4-Class2\sig_mat_1_0.3.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class4-Class2\sig_mat_1_0.3.xlsx --out_name sig_val_0.3"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class4-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class4-Class2\sig_mat_1_0.2.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class4-Class2\sig_mat_1_0.2.xlsx --out_name sig_val_0.2"
        ,"networks_correlations_code\\networks_correlations\\statistics\\compare_mat.py --out_folder Y:\Vicki\TwoGroups\Version1\Class4-Class2 --mat1 Y:\Vicki\TwoGroups\Version1\Group1\Class4-Class2\sig_mat_1_0.1.xlsx --mat2 Y:\Vicki\TwoGroups\Version1\Group2\Class4-Class2\sig_mat_1_0.1.xlsx --out_name sig_val_0.1"        
        ]
        
    for command in compare_mat_commands:
        print(command)
        return_code = subprocess.call(command, shell=True)
        checkSubprocessRet(return_code)
       