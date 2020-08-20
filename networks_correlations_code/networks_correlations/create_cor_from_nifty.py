import subprocess, argparse, os
import time


if __name__ == "__main__":
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--preproc_folder', required=True, type=str, help='path to folder containing all the preprocessed scans')
	parser.add_argument('--atlas_coords', required=True, type=str, help='path to text file with all the atlas coordinates')
	parser.add_argument('--out_folder', required=True, type=str, help='output folder')
	parser.add_argument('--not_create_subjects', action='store_true')
	parser.add_argument('--networks', required=False, nargs='+', help='Plot the common matrices for some networks, if you want the combination of every two networks use --networks all')
	parser.add_argument('--min_r', required=False, type=float, default = 0.7, help='Minumun R value for brain connection plotting')
	args = parser.parse_args()
	if (args.not_create_subjects == False):
		create_subject_dict_command = "create_subjects_dict.py --preproc_folder " + args.preproc_folder + " --atlas_coords " + args.atlas_coords + " --out_folder " + args.out_folder
		print(create_subject_dict_command)
		subprocess.call(create_subject_dict_command, shell=True)
	subject_dict_path = args.out_folder + "/subjects_data.pkl"
	run_common_mat_command = "run_common_mat.py --out_folder " + args.out_folder + " --subjects_data_dict " +  subject_dict_path
	print(run_common_mat_command)
	if args.networks != None:
		networks_str = ""
		for network in args.networks:
			networks_str = networks_str + str(network) + " "
		run_common_mat_command = run_common_mat_command + " --networks " + networks_str + " --atlas " + args.atlas_coords + " --min_r " + str(args.min_r)
	subprocess.call(run_common_mat_command, shell=True)
	
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	