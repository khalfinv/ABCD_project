import torch
import torch.nn as nn
import torch.utils.data as data
import os, sys, pickle
import math


# define a model:
class DeepFCNet(nn.Module):
	def __init__(self):
		super(DeepFCNet, self).__init__()
		self.feature_extractor = FeatureExtractor(375,32)
		self.similarity_measure = SimilarityMeasureNetwork()
		self.classification_net = ClassificationNet(34716,3)
		
		
	def forward(self, x):
		print ("x:", x.size())
		subjects = torch.zeros(0)
		for subject in x:
			out = self.feature_extractor(subject[0])
			out = torch.unsqueeze(out,dim=0)
			for net in subject[1:]:
				new_net = self.feature_extractor(net)
				new_net = torch.unsqueeze(new_net,dim=0)				
				out = torch.cat([out,new_net],dim=0)
			print ("out:", out.size())
			fc_all = torch.zeros(0)
			for net1_index in range(out.size(0)):
				for net2_index in range(net1_index+1,out.size(0)):
					two_nets=torch.cat([out[net1_index],out[net2_index]],dim=0)
					#print("two_nets",two_nets.size())
					fc_out=self.similarity_measure(two_nets)
					fc_all = torch.cat([fc_all,fc_out],dim=0)
			#print("fc_all:", fc_all)
			fc_all = torch.unsqueeze(fc_all,dim=0)			
			subjects = torch.cat([subjects,fc_all],dim=0)
			print("subjects",subjects.size())
		return self.classification_net(subjects)
		
class FeatureExtractor(nn.Module):
	def __init__(self, input_size, num_classes):
		super(FeatureExtractor, self).__init__()
		self.fc1 = nn.Linear(input_size, 64)
		self.relu = nn.ReLU()
		self.sig = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.rrelu = nn.RReLU()
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, num_classes)
		self.dropout = nn.Dropout(p=0.2)
		self.bn1 = nn.BatchNorm1d(784)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(32)

	def forward(self, x):
		out = self.fc1(x)
		out = self.rrelu(out)
		out = self.dropout(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out
class ClassificationNet(nn.Module):
	def __init__(self, input_size, num_classes):
		super(ClassificationNet, self).__init__()
		self.fc1 = nn.Linear(input_size, 4096)
		self.fc2 = nn.Linear(4096, 256)
		self.fc3 = nn.Linear(256, 64)
		self.fc4 = nn.Linear(64, num_classes)
		self.relu = nn.ReLU()
		self.sig = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.rrelu = nn.RReLU()
		self.dropout = nn.Dropout(p=0.2)
		self.logsoftmax = nn.LogSoftmax(dim=1)


	def forward(self, x):
		out = self.fc1(x)
		out = self.rrelu(out)
		out = self.dropout(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		out = self.relu(out)
		out = self.fc4(out)
		return self.logsoftmax(out)
		

		
class SimilarityMeasureNetwork(nn.Module):
	def __init__(self):
		super(SimilarityMeasureNetwork, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(64, 32),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(32, 16),
			nn.ReLU())
		self.fc3 = nn.Sequential(
			nn.Linear(16, 8),
			nn.ReLU())
		self.fc4 = nn.Sequential(
			nn.Linear(8, 1),
			nn.Tanh())
			
	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		#print("SimilarityMeasureNetwork out", out)
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
    device = torch.device("cuda:1")
    if use_gpu:
        x = x.to(device)
    return x
	
