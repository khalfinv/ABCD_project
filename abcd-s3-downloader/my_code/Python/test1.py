
import pandas as pd

if __name__ == '__main__':

	df_behav_data = pd.read_excel("C:\\Users\\skhalfin.TD-ST\\Google Drive\\secondDegree\\ABCD_important_data\\profile_data_no_brain_corr.xlsx")
	df_subject_keys = pd.read_csv("C:\\Users\\skhalfin.TD-ST\\Desktop\\ABCD_project\\abcd-s3-downloader\\Connectivity_Matrix\\derivatives\\abcd-hcp-pipeline\\subjects.csv")
	merged = pd.merge(left=df_behav_data, right=df_subject_keys, left_on='SUBJECTKEY', right_on='SUBJECTKEY')
	print(merged.shape)
	merged.to_csv("behav_data.csv", index=False)