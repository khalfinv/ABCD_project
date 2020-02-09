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



if __name__ == "__main__":

	# Hyper Parameters
	num_epochs = 15
	batch_size = 5
	learning_rate = 0.001


	parser = argparse.ArgumentParser()
	parser.add_argument('--data_folder', required=True, help='path to folder contaning all the data - train, validate and test')
	#parser.add_argument('--out_folder', required=True, help='path to folder where to save the model')
	args = parser.parse_args()


	train_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"train_set_class.pkl"))
		
	validate_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"validate_set_class.pkl"))


	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=True)
	DeepFCNet = common.DeepFCNet()

	criterion = nn.NLLLoss()
	optimizer = torch.optim.Adam(DeepFCNet.parameters(),lr = learning_rate)
	DeepFCNet.train() # turning the network to training mode, affect dropout and batch-norm layers if exists
	outputs_all = torch.zeros(0)
	scores_all = torch.zeros(0).float()
	for i, (time_series, scores) in enumerate(train_loader):
		#time_series = time_series.unsqueeze(1)
		time_series = common.to_cuda(time_series)
		scores = common.to_cuda(scores)
		
		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = DeepFCNet(time_series)
		loss = criterion(outputs, scores)
		loss.backward()
		optimizer.step()
		print("loss",loss.item())
		print("outputs", outputs.size())
		_, predicted = torch.max(outputs, 1)
		print("predicted:", predicted)
		print("scores:", scores)
		err = (predicted.cpu() != scores.cpu()).sum()
		print("err:", err)