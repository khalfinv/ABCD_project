#!/usr/bin/python3
import os, sys, argparse, pickle
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, type=str, help='path to pkl file with data_set')
	args = parser.parse_args()

	#open and read the pkl file of train-validate-test dictionaries
	pkl_file = open(args.dataset, 'rb')
	subjects = pickle.load(pkl_file)
	X = subjects["X"]
	y = subjects["y"]
	pkl_file.close()
	num_of_below_avg = 0
	num_of_avg = 0
	num_of_above_avg = 0
	for score in y:
		if score == 0:
			num_of_below_avg+=1
		elif score == 2:
			num_of_avg+=1
		else:
			num_of_above_avg+=1

	print ("**********statistics***********")
	print ("dataset size: ", len(y))
	print ("num of below average subjects: ", num_of_below_avg, " ", (num_of_below_avg/len(y))*100,"%")
	print ("num of average subjects: ", num_of_avg,  " ", (num_of_avg/len(y))*100,"%")
	print ("num of above average subjects: ", num_of_above_avg,  " ", (num_of_above_avg/len(y))*100,"%")