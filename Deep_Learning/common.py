import torch
import torch.nn as nn
import torch.utils.data as data
import os, sys, pickle


# define a model:
class ABCD_Net(nn.Module):
    def __init__(self):
        super(ABCD_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32))
        self.fc1 = nn.Sequential(
            nn.Linear(93*66*32, 1),
            nn.ReLU())
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
		
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
	if use_gpu:
		x = x.cuda()
	return x
	
