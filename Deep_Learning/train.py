import torch
import torch.nn as nn
import os, sys, argparse
import numpy as np
import torch.utils.data as data
import pandas as pd
import math
from sklearn.metrics import r2_score
import common
import matplotlib
import matplotlib.pyplot as plt

# Hyper Parameters
num_epochs = 5
batch_size = 20
learning_rate = 0.001


def save_checkpoint(model,filepath):
	state = {
	'state_dict': model.state_dict(),
	}
	torch.save(state, filepath)
	
def plotGraph(graphLabel, xValues, yValuesTrain, yValuesTest, xLabel, yLabel, outputFile):
	fig, ax = plt.subplots(1, 1, figsize=(6, 5))
	plt.title(graphLabel)
	trainPlot, = plt.plot(xValues, yValuesTrain, label="Train")
	testPlot, = plt.plot(xValues, yValuesTest, label="Evaluate")
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	plt.legend(handles=[trainPlot, testPlot])
	plt.savefig(outputFile)
	
def trainFunc(net, train_loader, criterion, optimizer):
	lossSum = 0 # sum of all loss 
	net.train() # turning the network to training mode, affect dropout and batch-norm layers if exists
	outputs_all = torch.zeros(0)
	scores_all = torch.zeros(0).float()
	for i, (time_series, scores) in enumerate(train_loader):
		time_series = time_series.unsqueeze(1)
		time_series = common.to_cuda(time_series)
		scores = common.to_cuda(scores)
		
		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = net(time_series)
		scores = scores.view(-1,1)
		loss = criterion(outputs, scores.float())
		loss.backward()
		optimizer.step()
		lossSum += loss.item()
		outputs_all = torch.cat([outputs_all, outputs.reshape(-1).cpu()])
		scores_all = torch.cat([scores_all, scores.float().cpu()])
		
		if (i+1) % 5 == 0:
			print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
				   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
	r_squared = r2_score(scores_all.tolist(),outputs_all.tolist()) 
	return (lossSum / i), r_squared 

def evaluateFunc(net, validate_loader, criterion):
	lossSum = 0 # sum of all loss 
	net.eval() # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
	outputs_all = torch.zeros(0)
	scores_all = torch.zeros(0).float()
	for i, (time_series, scores) in enumerate(validate_loader):
		time_series = time_series.unsqueeze(1)
		time_series = common.to_cuda(time_series)
		scores = common.to_cuda(scores)
		outputs = net(time_series)
		scores = scores.view(-1,1)
		loss = criterion(outputs, scores.float())
		lossSum += loss.item()
		outputs_all = torch.cat([outputs_all, outputs.reshape(-1).cpu()])
		scores_all = torch.cat([scores_all, scores.float().cpu()])
	r_squared = r2_score(scores_all.tolist(),outputs_all.tolist()) 
	return (lossSum / i), r_squared	
	

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_folder', required=True, help='path to folder contaning all the data - train, validate and test')
	parser.add_argument('--out_folder', required=True, help='path to folder where to save the model')
	args = parser.parse_args()


	train_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"train_set.pkl"))
		
	validate_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"validate_set.pkl"))


	time_series, _, = train_dataset[10]
	print (time_series.size())

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=True)
											  
	validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
										   batch_size=batch_size,
										   shuffle=False)
										   
											   
	#create object of the ABCD_Net class
	net = common.ABCD_Net()
	net = common.to_cuda(net)
	# Loss and Optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
	trainLossArr = []
	trainErrArr = []
	evaluateLossArr = []
	evaluateErrArr = []
	#Iterate num_epoch times and in each epoch train on total data amount / batch_size
	for epoch in range(num_epochs):
		trainLoss, trainErr = trainFunc(net, train_loader, criterion, optimizer)
		evaluateLoss, evaluateErr  = evaluateFunc(net, validate_loader, criterion)
		
		trainLossArr.append(trainLoss)
		trainErrArr.append(trainErr)
		evaluateLossArr.append(evaluateLoss)
		evaluateErrArr.append(evaluateErr)
	
	#Save the Model
	save_checkpoint(net, os.path.join(args.out_folder, "model.pkl"))
	
	#plot graphs
	#err vs epoch
	plotGraph('Error vs Epoch', range(num_epochs), trainErrArr, evaluateErrArr, 'Epoch', 'R-Square', 'rSquarePlot.png')
	#loss vs epoch
	trainLossArr = [round(loss,3) for loss in trainLossArr]
	testLossArr = [round(loss,3) for loss in evaluateLossArr]
	plotGraph('Loss vs Epoch', range(num_epochs), trainLossArr, evaluateLossArr, 'Epoch', 'Loss', 'lossPlot.png')


	
