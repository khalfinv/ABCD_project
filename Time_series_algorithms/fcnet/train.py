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
import time

# Hyper Parameters
num_epochs = 30
batch_size = 50
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
    for i, (time_series, scores) in enumerate(train_loader):
        time_series = common.to_cuda(time_series)
        scores = common.to_cuda(scores)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(time_series)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        lossSum += loss.item()
        total += scores.size(0)
        _, predicted = torch.max(outputs, 1)
        errSum += (predicted.cpu() != scores.cpu()).sum()
        
        if (i+1) % 30 == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
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
    for i, (time_series, scores) in enumerate(validate_loader):
        time_series = common.to_cuda(time_series)
        outputs = net(time_series)
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

    train_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"train_set_class.pkl"))
        
    validate_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"validate_set_class.pkl"))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
                                              
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
										   
											   
    #create object of the fMRI_CNN class
    DeepFCNet = common.DeepFCNet()
    DeepFCNet = common.to_cuda(DeepFCNet)
    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(DeepFCNet.parameters(),lr = learning_rate)
    trainLossArr = []
    trainErrArr = []
    evaluateLossArr = []
    evaluateErrArr = []
    #Iterate num_epoch times and in each epoch train on total data amount / batch_size
    for epoch in range(num_epochs):
        trainLoss, trainErr = trainFunc(DeepFCNet, train_loader, criterion, optimizer)
        evaluateLoss, evaluateErr  = evaluateFunc(DeepFCNet, validate_loader, criterion)
        trainLossArr.append(trainLoss)
        trainErrArr.append(trainErr)
        evaluateLossArr.append(evaluateLoss)
        evaluateErrArr.append(evaluateErr)

    #Save the Model
    save_checkpoint(DeepFCNet, os.path.join(args.out_folder, "DeepFCNet_model.pkl"))

    #plot graphs
    #err vs epoch
    plotGraph('Error vs Epoch', range(num_epochs), trainErrArr, evaluateErrArr, 'Epoch', 'Error', os.path.join(args.out_folder, "errPlot.png"))
    #loss vs epoch
    trainLossArr = [round(loss,3) for loss in trainLossArr]
    testLossArr = [round(loss,3) for loss in evaluateLossArr]
    plotGraph('Loss vs Epoch', range(num_epochs), trainLossArr, evaluateLossArr, 'Epoch', 'Loss', os.path.join(args.out_folder, "lossPlot.png"))


	
