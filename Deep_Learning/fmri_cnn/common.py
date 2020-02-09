import torch
import torch.nn as nn
import torch.utils.data as data
import os, sys, pickle


# define a model:
class fMRI_CNN(nn.Module):
	def __init__(self):
		super(fMRI_CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm2d(8))
		self.layer2 = nn.Sequential(
			nn.Conv2d(8, 8, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm2d(8))
		self.layer3 = nn.Sequential(
			nn.Conv2d(8, 8, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm2d(8),
			nn.MaxPool2d(2))
		self.layer4 = nn.Sequential(
			nn.Conv2d(8, 16, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm2d(16))
		self.layer5 = nn.Sequential(
			nn.Conv2d(16, 16, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm2d(16))
		self.layer6 = nn.Sequential(
			nn.Conv2d(16, 16, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm2d(16),
			nn.MaxPool2d(2))
		self.layer7 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm2d(32))
		self.layer8 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm2d(32))
		self.layer9 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm2d(32),
			nn.Dropout(p=0.3))
		self.fc1 = nn.Sequential(
			nn.Linear(83*55*32, 14608),
			nn.LeakyReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(14608, 1460),
			nn.LeakyReLU())
		self.fc3 = nn.Sequential(
			nn.Linear(1460, 3),
			nn.LeakyReLU())
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.layer6(out)
		out = self.layer7(out)
		out = self.layer8(out)
		out = self.layer9(out)
		print ("out", out.size())
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = self.fc2(out)
		out = self.fc3(out)
		return self.logsoftmax(out)
		
class TimeseriesDataset(data.Dataset):
	def __init__(self, dataset_path):
		self.time_series = []
		self.scores = []
		
		pkl_file = open(dataset_path, 'rb')
		self.time_series, self.scores = pickle.load(pkl_file)
		pkl_file.close()
		
		self.time_series = [torch.from_numpy(ts) for ts in self.time_series]
				
	def __len__(self):
		return len(self.time_series)

	def __getitem__(self, idx):
		time_series, scores = self.time_series[idx], self.scores[idx]
		return time_series, scores
		
#This function enable the model to run in cpu and gpu	
def to_cuda(x):
    use_gpu = torch.cuda.is_available()	
    device = torch.device("cuda:0")
    if use_gpu:
        x = x.to(device)
    return x

	
