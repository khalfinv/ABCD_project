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

input_size = 169
		   
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

	
def testFunc(net, test_loader): 
	"""Test the network
	param net: nn.Module. The network to test
	param test_loader: data.Dataset. The test dataset

	return: accuracy rate
	"""
	total = 0 # sum of total scores 
	correct = 0
	net.eval() # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
	num_of_below_avg = 0
	num_of_avg = 0
	num_of_above_avg = 0
	nb_classes = 3
	conf_matrix = torch.zeros(nb_classes, nb_classes)
	for i, (correlations, scores) in enumerate(test_loader):
		correlations = common.to_cuda(correlations)
		outputs = net(correlations)
		_, predicted = torch.max(outputs, 1)
		if(predicted == 0):
			num_of_below_avg+=1
		elif(predicted == 1):
			num_of_avg+=1
		else:
			num_of_above_avg+=1
		for t, p in zip(scores, predicted.cpu()):
			conf_matrix[t, p] += 1
		total += scores.size(0)
		correct += (predicted.cpu() == scores).sum()
		TP = conf_matrix.diag()
	print(conf_matrix)
	for c in range(nb_classes):
		idx = torch.ones(nb_classes).byte()
		idx[c] = 0
		# all non-class samples classified as non-class
		TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
		# all non-class samples classified as class
		FP = conf_matrix[idx, c].sum()
		# all class samples not classified as class
		FN = conf_matrix[c, idx].sum()
	accuracy = (100*float(correct)/total)
	print("number of below:", num_of_below_avg, " number of average:", num_of_avg, " number of above:", num_of_above_avg)
	print (" True Negative: ", TN, " ", (TN/total)*100,"%"" False Positive: ", FP, " ", (FP/total)*100,"%", "False Negative: ", FN , " ", (FN/total)*100, "%")
	return accuracy
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_folder', required=True, help='path to folder contaning all the data - train, validate and test')
	parser.add_argument('--model', required=True, help='path to the model after training')
	args = parser.parse_args()
	
	test_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"test_set_class.pkl"))
	
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											   shuffle=False)
	net = common.corr_nn(input_size, 3)
	#load the model						   
	net = load_checkpoint(net, args.model)
	net = common.to_cuda(net)
	# Test the Model
	testErr = testFunc(net, test_loader)
	print (testErr)
