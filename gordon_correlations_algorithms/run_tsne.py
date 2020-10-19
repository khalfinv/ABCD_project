#Run TSNE

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

if __name__ == "__main__":
	data_df = pd.read_csv("all_data.csv")
	DMN_cols = ["RSFMRI_C_NGD_CGC_NGD_DT","RSFMRI_C_NGD_DT_NGD_DT","RSFMRI_C_NGD_DT_NGD_DLA","RSFMRI_C_NGD_DT_NGD_FO",
						"RSFMRI_C_NGD_DT_NGD_SA", "RSFMRI_C_NGD_DT_NGD_VTA"]
	CO_cols = ["RSFMRI_C_NGD_CGC_NGD_CGC","RSFMRI_C_NGD_CGC_NGD_DT","RSFMRI_C_NGD_CGC_NGD_DLA","RSFMRI_C_NGD_CGC_NGD_FO",
				"RSFMRI_C_NGD_CGC_NGD_SA", "RSFMRI_C_NGD_CGC_NGD_VTA"]
	DAN_cols = ["RSFMRI_C_NGD_CGC_NGD_DLA", "RSFMRI_C_NGD_DT_NGD_DLA", "RSFMRI_C_NGD_DLA_NGD_DLA", "RSFMRI_C_NGD_DLA_NGD_FO", 
					"RSFMRI_C_NGD_DLA_NGD_SA", "RSFMRI_C_NGD_DLA_NGD_VTA"]
	FP_col = ["RSFMRI_C_NGD_CGC_NGD_FO", "RSFMRI_C_NGD_DT_NGD_FO", "RSFMRI_C_NGD_DLA_NGD_FO", "RSFMRI_C_NGD_FO_NGD_FO",
				"RSFMRI_C_NGD_FO_NGD_SA", "RSFMRI_C_NGD_FO_NGD_VTA"]
	Salience_cols = ["RSFMRI_C_NGD_DT_NGD_SA","RSFMRI_C_NGD_CGC_NGD_SA","RSFMRI_C_NGD_DLA_NGD_SA","RSFMRI_C_NGD_FO_NGD_SA",
						"RSFMRI_C_NGD_SA_NGD_SA", "RSFMRI_C_NGD_SA_NGD_VTA"]
	VAN_cols = ["RSFMRI_C_NGD_DT_NGD_VTA","RSFMRI_C_NGD_CGC_NGD_VTA","RSFMRI_C_NGD_DLA_NGD_VTA","RSFMRI_C_NGD_FO_NGD_VTA",
						"RSFMRI_C_NGD_SA_NGD_VTA", "RSFMRI_C_NGD_VTA_NGD_VTA"]
	#for i in [2,3]:
		#indexNames = data_df[ (data_df['label'] == 1) | (data_df['label'] == i)].index
		#data_df_new = data_df.loc[indexNames]
	X = data_df.drop('label', axis = 1)
	for pair in itertools.combinations(X.columns, r=3):
		print(list(pair))
		X_new = X[list(pair)]
		print(X_new.values.shape)
		tsne_results = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200).fit_transform(X_new.values)
		print(tsne_results.shape)
		data_df['tsne-2d-one'] = tsne_results[:,0]
		data_df['tsne-2d-two'] = tsne_results[:,1]
		plt.figure(figsize=(16,10))
		sns.scatterplot(
			x='tsne-2d-one', y='tsne-2d-two',
			hue="label",
			palette=sns.color_palette("hls", 4),
			data=data_df,
			legend="full",
			alpha=0.3
		)
		plt.savefig("TSNE_output\\tsne_graph_" + str(pair)+".png")
		plt.clf()