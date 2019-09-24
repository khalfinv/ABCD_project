import subprocess, os, sys, pickle, argparse

if __name__ == "__main__":
	#parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_pkl', required=True, type=str, help='full path to t1 to fmri dictionary pkl file')
    args = parser.parse_args()
	
    dict_pkl = args.dict_pkl
	
	#read dict from pkl file
    pkl_file = open(dict_pkl, 'rb')
    dict = pickle.load(pkl_file)
    pkl_file.close()
	
    num_of_part1 = 1600
    num_of_part2 = len(dict) - num_of_part1
	
    part1_dict = {k: dict[k] for k in sorted(dict.keys())[:num_of_part1]}
    part2_dict = {k: dict[k] for k in sorted(dict.keys())[-num_of_part2:]}
				
    #print ("actual_dict: \n ", sorted(dict))
    print ("actual_dict_size: ", len(dict))
    #print ("part1_dict: \n ", part1_dict)
    print ("part1_dict_size: ", len(part1_dict))
    #print ("part2_dict: \n ", part2_dict.keys())
    print ("part2_dict_size: ", len(part2_dict))
	
	#save to pkl files
    part1_output = open('part1_dict.pkl', 'wb')
    pickle.dump(part1_dict, part1_output)
    part1_output.close()

    part2_output = open('part2_dict.pkl', 'wb')
    pickle.dump(part2_dict, part2_output)
    part2_output.close()