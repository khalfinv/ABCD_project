#!/usr/bin/python3

import subprocess, argparse, os
import multiprocessing as mp
import test

def run_train_test(cv_data_folder, i):
	cv_path = os.path.join(cv_data_folder,("cv" + str(i+1)))
	subprocess.call("train.py --data_folder " + cv_path +  " --out_folder " + cv_path , shell=True)
	validate_set_path = os.path.join(cv_path,"validate_set.pkl")
	model_path = os.path.join(cv_path,"cogtests_nn_model.pkl")
	return test.testFunc(validate_set_path,model_path,cv_path)

def collect_errors(exception):
	""" Callback for errors collecting from threads. Get the exception and write to file
	param exception: Exception. The exception that was raised
	"""
	print(exception)
	
def collect_results(accuracy, f1_macro, f1_weighted):
	"""Collect the results from postProcessing function. 
	   Insert the result to allParticipantsDic.
	param result: dictionary raw. 
		key: subject's key . 
		value: {"time_series" : matrix of time series after censoring (time_points, power_rois), "covariance" : covariance matrix of power rois (power_rois, power_rois),
			"correlation" : correlation matrix of power rois (power_rois, power_rois), "num_of_volumes" : num of volumes left after censoring}
	return: None 
	"""
	global total_acc
	global total_f1_macro
	global total_f1_weighted
	total_acc+=accuracy
	total_f1_macro+=f1_macro
	total_f1_weighted+=f1_weighted
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--cv_data_folder', required=True, help='path to folder contaning all cross validation folders')
	args = parser.parse_args()
	K = 10
	total_acc = 0
	total_f1_macro = 0
	total_f1_weighted = 0
	pool = mp.Pool()
	for i in range(K):
		[pool.apply_async(run_train_test, args=(args.cv_data_folder, i,),callback=collect_results, error_callback = collect_errors)]
	pool.close() 
	pool.join()

		
	print("Average accuracy:", total_acc/K)
	print("Average f1 macro:", total_f1_macro/K)
	print("Average f1 weighted:", total_f1_weighted/K)

