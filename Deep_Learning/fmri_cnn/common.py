import torch
import torch.nn as nn
import torch.utils.data as data
import os, sys, pickle


# define a model:
class fMRI_CNN(nn.Module):
    def __init__(self):
        super(fMRI_CNN, self).__init__()
        self.connect = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p=0.3))
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256))
        self.fc1 = nn.Sequential(
            nn.Linear(11*41*256, 1000),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3))
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 3),
            nn.LeakyReLU())
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.connect(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.connect(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.connect(out)
        #print ("out", out.size())
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        return self.logsoftmax(out)
		
class TimeseriesDataset(data.Dataset):
	def __init__(self, dataset_path):
		self.time_series = []
		self.scores = []
		
		pkl_file = open(dataset_path, 'rb')
		self.subjects = pickle.load(pkl_file)
		pkl_file.close()
				
	def __len__(self):
		return len(self.subjects)

	def __getitem__(self, idx):
		time_series, score = self.subjects[idx]
		return torch.from_numpy(time_series), score
		
#This function enable the model to run in cpu and gpu	
def to_cuda(x):
    use_gpu = torch.cuda.is_available()	
    device = torch.device("cuda:0")
    if use_gpu:
        x = x.to(device)
    return x

	
