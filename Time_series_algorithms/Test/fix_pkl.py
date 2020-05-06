
import os, sys, pickle

if __name__ == "__main__":

	#open and read the pkl file of all subjects' data	
	pkl_file = open("matAndTimeSerias.pkl", 'rb')
	subjects_data_dict = pickle.load(pkl_file)
	pkl_file.close()
	
	fixed_dict = {}
	for key, val in subjects_data_dict.items():
		key = key.split("/")[10]
		fixed_dict[key]=val
		
    #Save
	fixed_pkl = open('matAndTimeSerias_fixed.pkl', mode="wb")
	pickle.dump(fixed_dict, fixed_pkl)
	fixed_pkl.close() 
	
	
	