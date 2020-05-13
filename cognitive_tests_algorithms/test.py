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

input_size = 14
accuracy = 0
num_classes = 4
		   
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

	
def testFunc(net, test_loader, out_folder): 
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
	nb_classes = 4
	conf_matrix = torch.zeros(nb_classes, nb_classes)
	for i, (correlations, scores) in enumerate(test_loader):
		correlations = common.to_cuda(correlations)
		outputs = net(correlations)
		_, predicted = torch.max(outputs, 1)
		if(predicted == 0):
			num_of_below_avg+=1
		elif(predicted == 1):
			num_of_above_avg+=1
		else:
			num_of_avg+=1
		total += scores.size(0)
		correct += (predicted.cpu() == scores).sum()
		for t, p in zip(scores, predicted.cpu()):
			conf_matrix[t, p] += 1
	TP = conf_matrix.diag()
	for c in range(nb_classes):
		idx = torch.ones(nb_classes).byte()
		idx[c] = 0
		# all non-class samples classified as non-class
		TN = float(conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()) #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
		# all non-class samples classified as class
		FP = float(conf_matrix[idx, c].sum())
		# all class samples not classified as class
		FN = float(conf_matrix[c, idx].sum())
	accuracy = (100*float(correct)/total)
	out_file = open(os.path.join(out_folder,"results.txt"),'w')
	out_file.write(str(conf_matrix.numpy()) + "\n")
	out_file.write("False Positive: " + str(FP) + " " + "%.3f" % ((FP/total)*100) + "%" + " False Negative: " + str(FN) + " " + "%.3f" % ((FN/total)*100) + "%" + "\n")
	out_file.write("accuracy: " + "%.3f" % accuracy + "\n")
	out_file.close()
	return accuracy
	
def main(test_dataset_path, model_path, out_folder ):
	
	test_dataset = common.CognitiveTestsDataset(test_dataset_path)
	
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											   shuffle=False)
	net = common.cogtests_nn(input_size, num_classes)
	#load the model						  
	net = load_checkpoint(net, model_path)
	net = common.to_cuda(net)
	# Test the Model
	accuracy = testFunc(net, test_loader, out_folder)
	print ("%.3f" % accuracy)
	return accuracy
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_dataset', required=True, help='path to test dataset')
	parser.add_argument('--model', required=True, help='path to the model after training')
	parser.add_argument('--out_folder', required=True, help='path to the out folder')
	args = parser.parse_args()
	
	main(args.test_dataset, args.model, args.out_folder)
