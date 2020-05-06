#!/usr/bin/python3
"""
==========================================================================================
Common functions and classes for training and testing scripts. Includes:
- corr_nn network class
- fMRI time series dataset class
- to_cuda function
==========================================================================================

"""

import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn import preprocessing
import os, sys, pickle
import numpy as np





class corr_nn(nn.Module):
	def __init__(self, input_size, num_classes):
		super(corr_nn, self).__init__()
		self.fc1 = nn.Linear(input_size, 64)
		self.relu = nn.ReLU()
		self.sig = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.rrelu = nn.RReLU()
		self.leakyRelu = nn.LeakyReLU()
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, num_classes)
		self.dropout = nn.Dropout(p=0.2)
		self.bn1 = nn.BatchNorm1d(64)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		out = self.fc1(x)
		out = self.dropout(out)
		out = self.leakyRelu(out)
		out = self.fc2(out)
		out = self.dropout(out)
		out = self.leakyRelu(out)
		out = self.fc3(out)
		return out
		
class TimeseriesDataset(data.Dataset):
	def __init__(self, dataset_path):
		pkl_file = open(dataset_path, 'rb')
		subjects = pickle.load(pkl_file)
		res = list(zip(*subjects))
		self.x = res[0]
		self.y = res[1]	
		pkl_file.close()
				
	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		corr = self.x[idx]
		score = self.y[idx]
		return torch.FloatTensor(corr), score
		
	
def to_cuda(x):
    """This function enable the model to run in cpu and gpu.
    param x: The object that need to be copy to gpu if exists.
    return:  The input,x, after copied to gpu if gpu available.
    """
    use_gpu = torch.cuda.is_available()	
    device = torch.device("cuda:0")
    if use_gpu:
        x = x.to(device)
    return x

	
