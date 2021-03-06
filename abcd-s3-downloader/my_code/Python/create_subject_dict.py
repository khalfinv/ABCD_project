
import pandas as pd
import os, sys, pickle, argparse
import numpy as np
from sklearn.covariance import LedoitWolf
from nilearn.connectome import ConnectivityMeasure
import time
import multiprocessing


def get_time_series(path_to_excel):
	return pd.read_csv(path_to_excel, header=None).to_numpy()
if __name__ == '__main__':
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--time_series_folder', required=True, type=str, help='root folder with all time_series')
	parser.add_argument('--corr_matrix_folder', required=True, type=str, help='root folder with all correlation matrices and censor parcels')
	parser.add_argument('--output', required=True, type=str, help='output folder')
	args = parser.parse_args()
	subjects_dic_censored_indexes = {}
	subjects_dic_final = {}

	print("Extract 10 min censored indexes")
	for subject_folder in os.listdir(args.corr_matrix_folder):
		if (subject_folder not in subjects_dic_censored_indexes):
			subject_folder_full = os.path.join(args.corr_matrix_folder,subject_folder)
			if(os.path.isdir(subject_folder_full)):
				for root, dirs, files in os.walk(subject_folder_full):
					for file in files:
						if file.endswith("10min_conndata-network_censor.txt"):
							full_path = os.path.join(subject_folder_full,"ses-baselineYear1Arm1\\func",file)
							with open(full_path) as censor_file:
								censor_flags = np.array(censor_file.read().split('\n')[:-1])
								num_of_points = len(censor_flags)
								indexes = np.array(range(num_of_points))
								censor_flags = censor_flags.astype(np.int)				
								left_indexes = indexes[censor_flags != 0]
								subjects_dic_censored_indexes[subject_folder] = {"left_indexes" : left_indexes}
		else:
			print(subject_folder, " allready exists in dictionary!!!")
			
	print("Extract time series and covariance matrices")
	for subject_folder in os.listdir(args.time_series_folder):
		if (subject_folder in subjects_dic_censored_indexes):
			subject_folder_full = os.path.join(args.time_series_folder,subject_folder)
			if(os.path.isdir(subject_folder_full)):
				full_path = os.path.join(subject_folder_full,"ses-baselineYear1Arm1\\func","time_series.csv")
				if(os.path.exists(full_path)):	
					subject_key = subject_folder.split('-')[1]
					subject_key = subject_key[:4] + '_' + subject_key[4:]
					time_series = multiprocessing.Pool(1).map(get_time_series, [full_path])[0]
					#time_series = get_time_series(full_path)
					left_indexes = subjects_dic_censored_indexes[subject_folder]["left_indexes"]
					if(len(left_indexes) < 750):
						print("ERROR: less than 10 minute scan!!!")
					num_of_parcels = 333
					#Do not include subcortical ROIs
					time_series = time_series[:num_of_parcels,list(left_indexes)]
					time_series = np.transpose(time_series)
					cov_measure = ConnectivityMeasure(cov_estimator=LedoitWolf(assume_centered=False, block_size=1000, store_precision=False), kind='covariance')
					cov = []
					cov = cov_measure.fit_transform([time_series])[0, :, :]
					subjects_dic_final[subject_key] = {"time_series" : time_series, "covariance" : cov}
				else:
					print(full_path, " does not exists!!!")
					
	print("number of censored indexes subjecs: ", len(subjects_dic_censored_indexes))
	print("number of final subjects: ", len(subjects_dic_final))
	#write subjects keys to csv file
	print("Write subjects keys to csv file")
	subjects_df_final = pd.DataFrame({"subjects_keys" : list(subjects_dic_final.keys())})
	subjects_df_final.to_csv(args.output + "/subjects_final.csv", index=False)
	subjects_df_censored = pd.DataFrame({"subjects_keys" : list(subjects_dic_censored_indexes.keys())})
	subjects_df_censored.to_csv(args.output + "/subjects_censored_indexes.csv", index=False)
	#write the subjects_dic to pkl file
	print("Write to pkl file")
	f = open(args.output + '/subjects_data.pkl', mode="wb")
	pickle.dump(subjects_dic_final, f)
	f.close()
	
	#print the exucation time
	end = time.time()
	timeInSeconds = (end - start)
	timeInMinutes = timeInSeconds / 60
	timeInHours = int(timeInMinutes / 60)
	print ("Total time : " + str(timeInHours) + " hours and " + str(int(timeInMinutes % 60)) + " minutes")	

	