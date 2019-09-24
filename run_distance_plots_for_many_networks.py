import subprocess
import networkToIndexDic
import itertools
import plot_distance_vs_score
import pandas as pd
import pickle

if __name__ == "__main__":
	score_file = "Z:\\Users\\Vicki\\ABCD\\BehavioralData\\abcd_tbss01_nihTbx_cognitiveTests.xlsx"
	score_key = "NIHTBX_FLANKER_AGECORRECTED"
	subjects_data_dict = "Y:\\Vicki\\ABCD_rs_preprocess\\subjects_data.pkl"
	common_data_dict = "Y:\\Vicki\\ABCD_rs_preprocess\\RiemannMeanCovAndCorMat.pkl"
	out_folder = "Y:\\Vicki\\ABCD_rs_preprocess\\riemean"
	
	#Read the excel file 
	scores_file = pd.read_excel(io=score_file)
	#check if columns exists in the excel files
	if('SUBJECTKEY' not in scores_file):
		print("'SUBJECTKEY' column does not exists in excel file")
		sys.exit()

	if(score_key not in scores_file):
		print(score_key, " column does not exist in excel file")
		sys.exit()

	#open and read the pkl file of all subjects' matrices	
	pkl_file = open(subjects_data_dict, 'rb')
	allMatDict = pickle.load(pkl_file)
	pkl_file.close()

	#open and read the pkl file of the common matrices	
	pkl_file = open(common_data_dict, 'rb')
	(common_cov_mat, common_cor_mat) = pickle.load(pkl_file)
	pkl_file.close()
	for network in networkToIndexDic.dic.keys():
		all_scores, all_cov_distances, all_corr_distances = plot_distance_vs_score.score_vs_distance([network], common_cov_mat,common_cor_mat,allMatDict, scores_file, score_key)
		plot_distance_vs_score.calc_correlation([network], score_key, all_cov_distances, all_corr_distances, all_scores, out_folder)
		
	for pair in itertools.combinations(networkToIndexDic.dic.keys(), r=2):
		all_scores, all_cov_distances, all_corr_distances = plot_distance_vs_score.score_vs_distance(list(pair), common_cov_mat,common_cor_mat,allMatDict, scores_file, score_key)
		plot_distance_vs_score.calc_correlation(list(pair), score_key, all_cov_distances, all_corr_distances, all_scores, out_folder)		

	for tripel in itertools.combinations(networkToIndexDic.dic.keys(), r=3):
		all_scores, all_cov_distances, all_corr_distances = plot_distance_vs_score.score_vs_distance(list(tripel), common_cov_mat,common_cor_mat,allMatDict, scores_file, score_key)
		plot_distance_vs_score.calc_correlation(list(tripel), score_key, all_cov_distances, all_corr_distances, all_scores, out_folder)