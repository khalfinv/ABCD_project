#!/usr/bin/python3

import subprocess, argparse, os
import multiprocessing as mp
import test

def run_train_test(cv_data_folder, i):
	cv_path = os.path.join(cv_data_folder,("cv" + str(i+1)))
	subprocess.call("train.py --data_folder " + cv_path +  " --out_folder " + cv_path , shell=True)
	validate_set_path = os.path.join(cv_path,"validate_set.pkl")
	model_path = os.path.join(cv_path,"cogtests_nn_model.pkl")
	return test.main(validate_set_path,model_path,cv_path)

def collect_errors(exception):
	""" Callback for errors collecting from threads. Get the exception and write to file
	param exception: Exception. The exception that was raised
	"""
	print(exception)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--cv_data_folder', required=True, help='path to folder contaning all cross validation folders')
	args = parser.parse_args()
	K = 10
	total_acc = 0
	for i in range(K):
		total_acc+=run_train_test(args.cv_data_folder, i)
		
	print("Average accuracy:", total_acc/K)

