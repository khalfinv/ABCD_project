
import pandas as pd

if __name__ == '__main__':

	df_behav_data = pd.read_excel("C:\\Users\\skhalfin.TD-ST\\Google Drive\\secondDegree\\ABCD_important_data\\profile_behav_data.xlsx")
	df_subject_keys1 = pd.read_csv("Y:\\Vicki\\ABCD_derivatives\\subjects_final.csv")
	df_subject_keys2 = pd.read_csv("Y:\\Vicki\\ABCD_derivatives\\Connectivity_Matrix\\derivatives\\abcd-hcp-pipeline\\subjects10min.csv")
	for index, row in df_subject_keys1.iterrows():
		subject_key = row["SUBJECTKEY"]
		subject_key = subject_key.split('-')[1]
		subject_key = subject_key[:4] + '_' + subject_key[4:]
		row["SUBJECTKEY"] = subject_key
	subject_keys_merge = pd.merge(left=df_subject_keys1, right=df_subject_keys2, left_on='SUBJECTKEY', right_on='SUBJECTKEY')
	print("Subject key after merge:", subject_keys_merge.shape)
	subject_keys_merge = subject_keys_merge.dropna()
	print("Subject key after merge after null omitting:", subject_keys_merge.shape)
	merged = pd.merge(left=df_behav_data, right=subject_keys_merge, left_on='SUBJECTKEY', right_on='SUBJECTKEY')
	print("Before null ominting:", merged.shape)
	merged = merged.dropna()
	print("After null ominting:", merged.shape)
	merged.to_csv("behav_data.csv", index=False)