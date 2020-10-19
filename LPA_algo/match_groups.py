import pandas as pd
import os, sys, pickle, argparse, re
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests

def chisq_and_posthoc_corrected(df):
	print(chi2_contingency(df))
	# gathering all combinations for post-hoc chi2
	all_index_combinations = list(combinations(df.index, 2))
	all_columns_combinations = list(combinations(df.columns,2))
	print (all_columns_combinations)
	print("Significance results:")
	p_vals = []
	combinations_names = []
	for index_comb in all_index_combinations:
		# subset df into a dataframe containing only the pair "comb"
		new_df = df[(df.index == index_comb[0]) | (df.index == index_comb[1])]
		for col_comb in all_columns_combinations:
			new_df1 = new_df[[col_comb[0],col_comb[1]]]
			print (new_df1)
			# running chi2 test
			chi2, p, dof, ex = chi2_contingency(new_df1, correction=False)
			p_vals.append(p)
			combinations_names.append([index_comb,col_comb])
	reject_list, corrected_p_vals = multipletests(p_vals, method='bonferroni')[:2]
	print("groups\toriginal p-value\tcorrected p-value\treject?")
	for p_val, corr_p_val, reject, names in zip(p_vals, corrected_p_vals, reject_list, combinations_names):
		print(names, "\t", p_val, "\t", corr_p_val, "\t", reject,)
def create_pivot(df, group1_name, group2_name, group1_subs_names, group2_subs_names):
	grouped = df.groupby(by = [group1_name,group2_name]).count()
	df_pivot = grouped.reset_index()
	df_pivot = df_pivot.pivot(index=group1_name, columns=group2_name, values='SUBJECTKEY')
	df_pivot.index=group1_subs_names
	df_pivot.columns=group2_subs_names
	return df_pivot
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--behav_excel', required=True, type=str, help='path to excel file containing behaviotal groups')
	parser.add_argument('--brain_excel', required=True, type=str, help='path to excel file containing the brain networks groups')
	parser.add_argument('--num_groups', required=True, type=int, help='number of groups')
	parser.add_argument('--out_folder', required=False, type=str, default=".", help='path to output folder. Default is current folder')
	args = parser.parse_args()
	
	#Read the excel files 
	df_behav = pd.read_excel(io=args.behav_excel)
	df_brain = pd.read_excel(io=args.brain_excel)
	df_behav = df_behav[['SUBJECTKEY'] + ['Class']].rename(columns={'Class': 'Class_behav'})
	df_brain = df_brain[['SUBJECTKEY'] + ['Class']].rename(columns={'Class': 'Class_brain'})
	merge_df = pd.merge(df_behav,df_brain, left_on='SUBJECTKEY', right_on='SUBJECTKEY')
	behav_groups = ['emotion_inhibition', 'control','adhd','cognitive_inhibition']
	brain_groups = ['Group1', 'group2','group3','group4','group5','group6']
	
	
	#Behaviours data distributed in brain data
	new_df = create_pivot(merge_df, 'Class_brain', "Class_behav", brain_groups, behav_groups)
	chisq_and_posthoc_corrected(new_df)

				
	# #brain data distributed in brain data
	# new_df = create_pivot(merge_df, 'Class_behav', 'Class_brain', behav_groups, brain_groups)
	# chisq_and_posthoc_corrected(new_df)


	#create pai graphs
	for i in range(args.num_groups):
		new_df = merge_df[merge_df['Class_brain'] == (i+1)]
		new_df.groupby(['Class_behav']).sum().plot(kind='pie', y='Class_brain',autopct='%1.1f%%')
		print(len(new_df))
		plt.title(brain_groups[i])
		plt.legend(behav_groups);
		plt.savefig(brain_groups[i])
	
	# for i in range(4):
		# new_df = merge_df[merge_df['Class_behav'] == (i+1)]
		# new_df.groupby(['Class_brain']).sum().plot(kind='pie', y='Class_behav',autopct='%1.1f%%')
		# print(len(new_df))
		# plt.title(behav_groups[i])
		# plt.legend(brain_groups);
		# plt.savefig(behav_groups[i])