#!/usr/bin/python3
"""
==========================================================================================
Run training and evaluation. Plot error and loss graphs and save the trained model.
==========================================================================================

"""


import torch
import torch.nn as nn
import os, sys, argparse
import torch.utils.data as data
import pandas as pd
import common
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Hyper Parameters
num_epochs = 150
batch_size = 20
learning_rate = 0.001

	
def save_checkpoint(model,filepath):
	"""Save the model
	param model: nn.Module. The torch model
	param filepath: String. Full path for saving
	
	return: None.
    """
	state = {
	'state_dict': model.state_dict(),
	}
	torch.save(state, filepath)
	
def plotGraph(graphLabel, xValues, yValuesTrain, yValuesTest, xLabel, yLabel, outputFile):
	"""Create graph and save it.
	param graphLabel: String. Graph's title
	param xValues: List. List of x values
	param yValuesTrain: List. List of y values for train dataset
	param yValuesTest: List. List of y values for test dataset
	param xLabel: String. Label for x values
	param yLabel: String. Label for y values
	param outputFile: String. Full path for the output file
	
	return: None.
    """
	fig, ax = plt.subplots(1, 1, figsize=(6, 5))
	plt.title(graphLabel)
	trainPlot, = plt.plot(xValues, yValuesTrain, label="Train")
	testPlot, = plt.plot(xValues, yValuesTest, label="Evaluate")
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	plt.legend(handles=[trainPlot, testPlot])
	plt.savefig(outputFile)

	
def trainFunc(net, train_loader, criterion, optimizer):
	"""Train the network
	param net: nn.Module. The network to train
	param train_loader: data.Dataset. The train dataset
	param criterion: Loss function
	param optimizer: optimization algorhitm
	
	return: average over batches loss, average over batches error rate.
    """
	lossSum = 0 # sum of all loss 
	errSum = 0 # sum of all error rates 
	total = 0 # sum of total scores 
	net.train() # turning the network to training mode, affect dropout and batch-norm layers if exists
	for i, (correlations, scores) in enumerate(train_loader):
		correlations = common.to_cuda(correlations)
		scores = common.to_cuda(scores)
		
		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = net(correlations)
		loss = criterion(outputs, scores)
		#print(loss)
		loss.backward()
		optimizer.step()
		lossSum += loss.item()
		total += scores.size(0)
		_, predicted = torch.max(outputs, 1)
		#print(predicted)
		errSum += (predicted.cpu() != scores.cpu()).sum()
		
		# if (i+1) % 120 == 0:
			# print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
				   # %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
	return ((lossSum / i), (100*float(errSum)/total)) 

def evaluateFunc(net, validate_loader, criterion):
	"""Evaluate the network
	param net: nn.Module. The network to evaluate
	param validate_loader: data.Dataset. The validate dataset
	param criterion: Loss function

	return: average over batches loss, average over batches error rate.
	"""
	lossSum = 0 # sum of all loss 
	errSum = 0 # sum of all error rates 
	total = 0 # sum of total scores 
	net.eval() # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
	for i, (correlations, scores) in enumerate(validate_loader):
		correlations = common.to_cuda(correlations)
		outputs = net(correlations)
		loss = criterion(outputs.cpu(), scores)
		lossSum += loss.item()
		_, predicted = torch.max(outputs, 1)		
		total += scores.size(0)
		errSum += (predicted.cpu() != scores).sum()
	#return the average loss and average error
	return ((lossSum / i), (100*float(errSum)/total)) 
	

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_folder', required=True, help='path to folder contaning all the data - train, validate and test')
	parser.add_argument('--out_folder', required=True, help='path to folder where to save the model')
	args = parser.parse_args()


	train_dataset = common.CognitiveTestsDataset(os.path.join(args.data_folder,"train_set.pkl"))
		
	validate_dataset = common.CognitiveTestsDataset(os.path.join(args.data_folder,"validate_set.pkl"))


	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=True)
											  
	validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
										   batch_size=batch_size,
										   shuffle=False)
										   
	#Explore the classes
	values, counts = np.unique(train_dataset.y, return_counts=True)
	num_classes = len(values)
	print("The classes: " + str(values))
	percentage = counts / len(train_dataset.X)
	print("Train: The frequency: " + str(counts) + " The percentage: " + str(percentage) + "\n")

	values, counts = np.unique(validate_dataset.y, return_counts=True)
	percentage = counts / len(validate_dataset.X)
	print("Validate: The frequency: " + str(counts) + " The percentage: " + str(percentage) + "\n")

	#Initiate the network
	input_size = len(train_dataset.X[0])									   
	#create object of the cogtests_nn class
	cogtests_nn = common.cogtests_nn(input_size, num_classes)
	cogtests_nn = common.to_cuda(cogtests_nn)
	# Loss and Optimizer
	#criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.34, 0, 0.66]))
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(cogtests_nn.parameters(),lr = learning_rate)
	trainLossArr = []
	trainErrArr = []
	evaluateLossArr = []
	evaluateErrArr = []
	#Iterate num_epoch times and in each epoch train on total data amount / batch_size
	for epoch in range(num_epochs):
		trainLoss, trainErr = trainFunc(cogtests_nn, train_loader, criterion, optimizer)
		evaluateLoss, evaluateErr  = evaluateFunc(cogtests_nn, validate_loader, criterion)
		
		trainLossArr.append(trainLoss)
		trainErrArr.append(trainErr)
		evaluateLossArr.append(evaluateLoss)
		evaluateErrArr.append(evaluateErr)

	#Save the Model
	save_checkpoint(cogtests_nn, os.path.join(args.out_folder, "cogtests_nn_model.pkl"))

	#plot graphs
	#err vs epoch
	plotGraph('Error vs Epoch', range(num_epochs), trainErrArr, evaluateErrArr, 'Epoch', 'Error', os.path.join(args.out_folder, "errPlot.png"))
	#loss vs epoch
	trainLossArr = [round(loss,3) for loss in trainLossArr]
	testLossArr = [round(loss,3) for loss in evaluateLossArr]
	plotGraph('Loss vs Epoch', range(num_epochs), trainLossArr, evaluateLossArr, 'Epoch', 'Loss', os.path.join(args.out_folder, "lossPlot.png"))


	
