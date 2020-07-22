#!/usr/bin/python3
"""
==========================================================================================
Load the trained model and test with test dataset
==========================================================================================

"""


import torch	
import argparse
import common
import os
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
import itertools

		   
def load_checkpoint(model,filepath):
	"""Load the model
	param model: nn.Module. The torch model
	param filepath: String. Full path from loading

	return: the model loaded.
	"""
	# "lambda" allows to load the model on cpu in case it is saved on gpu
	state = torch.load(filepath,lambda storage, loc: storage)
	model.load_state_dict(state['state_dict'])
	return model

def plot_confusion_matrix(cm, classes, out_file,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(out_file)
	
def testFunc(test_dataset_path, model_path, out_folder ):
	#Load the test data
	test_dataset = common.CorrelationDataset(test_dataset_path)
	#Open the output file						
	out_file = open(os.path.join(out_folder,"results.txt"),'w')
	#Explore the classes
	values, counts = np.unique(test_dataset.y, return_counts=True)
	num_classes = len(values)
	percentage = counts / len(test_dataset.X)
	out_file.write("The classes: " + str(values) + " The frequency: " + str(counts) + " The percentage: " + str(percentage) + "\n")
	#Initiate the network
	input_size = len(test_dataset.X[0])
	net = common.corr_nn(input_size, num_classes)
	#load the model						  
	net = load_checkpoint(net, model_path)
	net = common.to_cuda(net)
	# Test the Model
	net.eval() # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
	correlations = torch.FloatTensor(test_dataset.X)
	correlations = common.to_cuda(correlations)
	outputs = net(correlations)
	_, predicted = torch.max(outputs, 1)
	#Output the results metrics
	confusion_matrix = skm.confusion_matrix(test_dataset.y, predicted)
	plot_confusion_matrix(confusion_matrix,values,os.path.join(out_folder,'confusion matrix.png'))
	out_file.write("Confusion matrix: \n" + str(confusion_matrix) + "\n\n")
	accuracy = skm.accuracy_score(test_dataset.y, predicted)
	out_file.write(skm.classification_report(test_dataset.y, predicted))
	f1_score_macro = skm.f1_score(test_dataset.y, predicted, average = 'macro')
	f1_score_weighted = skm.f1_score(test_dataset.y, predicted, average = 'weighted')
	# print ("accuracy: %.3f " % accuracy)
	# print ("f1_score_macro: %.3f " % f1_score_macro)
	# print ("f1_score_weighted: %.3f  " % f1_score_weighted)
	return accuracy, f1_score_macro, f1_score_weighted
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_dataset', required=True, help='path to test dataset')
	parser.add_argument('--model', required=True, help='path to the model after training')
	parser.add_argument('--out_folder', required=True, help='path to the out folder')
	args = parser.parse_args()
	
	testFunc(args.test_dataset, args.model, args.out_folder)
