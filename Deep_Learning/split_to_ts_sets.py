import os, sys, pickle, argparse, re
import pandas as pd

def createTimeSeriesSets(dict, score_file, score_key):
	time_series = []
	scores = []
	for key,value in dict.items():
		time_series.append((value["time_series"][:375]).T)
		raw = score_file.loc[lambda df: score_file['SUBJECTKEY'] == key] #Get the raw from the excel that match the subject_key. The raw is from type pandas.series
		score = raw[score_key].values[0]
		scores.append(score)
	return time_series, scores
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dicts_pkl', required=True, type=str, help='path to pkl file with test-validat-train dictionaries')
	parser.add_argument('--score_file', required=True, type=str, help='path to excel file containing the scores')
	parser.add_argument('--score_key', required=True, type=str, help='name of the test score column')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()

	#open and read the pkl file of all subjects' matrices	
	pkl_file = open(args.dicts_pkl, 'rb')
	all_dicts = pickle.load(pkl_file)
	pkl_file.close()

	train_dict = all_dicts['train_dict']
	validate_dict = all_dicts['validate_dict']
	test_dict = all_dicts['test_dict']

	#Read the excel file 
	df = pd.read_excel(io=args.score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in df):
		print("'SUBJECTKEY' column does not exists in excel file")
		sys.exit()

	if(args.score_key not in df):
		print(args.score_key, " column does not exist in excel file")
		sys.exit()

	#change the SUBJECT_KEY values to match to SUBJECT_KEY format in the subjects_data_dict (without "_")
	for i, row in df.iterrows(): 
		df.at[i,'SUBJECTKEY'] = re.sub(r"[^a-zA-Z0-9]","",df.at[i,'SUBJECTKEY'])

	train_set = screateTimeSeriesSets(train_dict, df, args.score_key)
	validate_set = createTimeSeriesSets(validate_dict, df, args.score_key)
	test_set = createTimeSeriesSets(test_dict, df, args.score_key)
	
	print("Save to pkl")
	train_file = open(args.out_folder + '/train_set.pkl', mode="wb")
	pickle.dump(train_set, train_file)
	train_file.close() 

	validate_file = open(args.out_folder + '/validate_set.pkl', mode="wb")
	pickle.dump(validate_set, validate_file)
	validate_file.close() 

	test_file = open(args.out_folder + '/test_set.pkl', mode="wb")
	pickle.dump(test_set, test_file)
	test_file.close() 
