import subprocess, os, sys, pickle, argparse

if __name__ == "__main__":
	#parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_pkl', required=True, type=str, help='full path to t1 to fmri dictionary pkl file')
    parser.add_argument('--invalid_subjects', required=True, type=str, help='full path invalid subjects file')
    args = parser.parse_args()
	
    dict_pkl = args.dict_pkl
    invalid_subjects = args.invalid_subjects
	
	#read dict from pkl file
    pkl_file = open(dict_pkl, 'rb')
    dict = pickle.load(pkl_file)
    pkl_file.close()
	
	#open and read invalid subjects file
    with open(invalid_subjects) as f:
        lines = f.read().splitlines() 
    lines = [line.replace('_', '') for line in lines]
    valid_subjects_dict = {}
    invalid_subjects_dict = {}
    for key, value in dict.items():
        if key in lines:
            invalid_subjects_dict[key] = value
        else:
            valid_subjects_dict[key] = value
				
    #print ("valid_subjects_dict: \n ", valid_subjects_dict)
    print ("valid_subjects_dict_size: ", len(valid_subjects_dict))
    #print ("invalid_subjects_dict: \n ", invalid_subjects_dict)
    print ("invalid_subjects_dict_size: ", len(invalid_subjects_dict))
	
	#save to pkl files
    valid_subjects_output = open('valid_subjects_dict.pkl', 'wb')
    pickle.dump(valid_subjects_dict, valid_subjects_output)
    valid_subjects_output.close()
	
    invalid_subjects_output = open('invalid_subjects_dict.pkl', 'wb')
    pickle.dump(invalid_subjects_dict, invalid_subjects_output)
    invalid_subjects_output.close()