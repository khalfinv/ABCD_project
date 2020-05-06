import os, sys, argparse
import networkToIndexDic 
import pickle


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--networks', required=True, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
	parser.add_argument('--subjects_data_dict', required=True, type=str, help='path to pkl file containing the subjects data dictionary')
	args = parser.parse_args()
	
	
	pkl_file = open(args.subjects_data_dict, 'rb')
	subjects_dict = pickle.load(pkl_file)
	pkl_file.close()
	
	for key,val in  subjects_dict.items():
		time_series = val["time_series"]
		listToSlice = []
		for network in args.networks:
			if network in networkToIndexDic.dic:
				listToSlice = listToSlice + list(networkToIndexDic.dic[network])
			else:
				print ( "The " + network + " network does not exist!!!")
		time_series = time_series[:, listToSlice]
		val["time_series"] = time_series
		
	
	new_subjects_dict_file = open("subjects_dict_" + str(args.networks) + ".pkl", mode="wb")
	pickle.dump(subjects_dict, new_subjects_dict_file)
	new_subjects_dict_file.close() 
	
		