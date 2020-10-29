import imblearn
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, pickle, argparse
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE

def plot_data(x, y, labels,fig_name):
	plt.figure(figsize=(16,10))
	sns.scatterplot(
		x=x, y=y,
		hue=labels,
		palette=sns.color_palette("hls", 4),
		legend="full",
		alpha=0.3
	)
	plt.savefig(fig_name)
	plt.clf()

if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, type=str, help='path to pkl file containing all training data')
	args = parser.parse_args()
	pkl_file = open(args.dataset, 'rb')
	subjects = pickle.load(pkl_file)
	X = subjects["X"]
	y = subjects["y"]
	
	# summarize class distribution
	counter = Counter(y)
	print(counter)
	# # scatter plot of examples by class label
	# for label, _ in counter.items():
		# row_ix = np.where(np.array(y) == label)[0]
		# plt.scatter(np.array(X)[row_ix, 0], np.array(X)[row_ix, 1], label=str(label))
	# plt.legend()
	# plt.show()
	tsne_results = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200).fit_transform(X)
	plot_data(tsne_results[:,0], tsne_results[:,1], y, "before-smote.png")
	
	# transform the dataset
	oversample = SMOTE()
	X, y = oversample.fit_resample(X, y)
	# summarize the new class distribution
	counter = Counter(y)
	print(counter)
	tsne_results = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200).fit_transform(X)
	plot_data(tsne_results[:,0], tsne_results[:,1], y, "after-smote.png")
	