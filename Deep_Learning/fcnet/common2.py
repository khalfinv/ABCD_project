import torch
import torch.nn as nn
import torch.utils.data as data
import os, sys, pickle
import math
import time
import numpy as np
import copy


# define a model:
class DeepFCNet(nn.Module):
    def __init__(self):
        super(DeepFCNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor = to_cuda(self.feature_extractor)
        self.similarity_measure = SimilarityMeasureNetwork()
        self.similarity_measure = to_cuda(self.similarity_measure)
        self.classification_net = ClassificationNet(9045,3)
        self.classification_net = to_cuda(self.classification_net)
		
		
    def forward(self, x):
        print ("x:", x.shape)
        subjects = to_cuda(torch.zeros([30,9045], dtype=torch.float32, requires_grad=True))
        subject_idx = 0
        for subject in x:         
            #out = torch.zeros(0)
            subject = subject.view(135,1,375)
            out = self.feature_extractor(subject)
            #print("out", out.size())
            all_combinations = to_cuda(torch.zeros ([9045, 64], dtype=torch.float32, requires_grad=True))
            #print("all_combinations:", all_combinations)
            i = 0
            for net1_index in range(out.size(0)):
                for net2_index in range(net1_index+1,out.size(0)):
                    all_combinations[i][:32] = out[net1_index]
                    all_combinations[i][32:] = out[net2_index]
                    i+=1
            
            fc_all=self.similarity_measure(all_combinations)
            subjects[subject_idx] = fc_all.squeeze(1)
            subject_idx+=1
        print("subjects:",subjects.size())
        out_class = self.classification_net(subjects)
        return out_class
		
class FeatureExtractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv1d(1, 32, 3),
            nn.BatchNorm1d(32),
			nn.LeakyReLU(),
			nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
			nn.Conv1d(16, 64, 3),
            nn.BatchNorm1d(64),
			nn.LeakyReLU(),
			nn.MaxPool2d(2))
		self.layer3 = nn.Sequential(
			nn.Conv1d(32, 96, 3),
            nn.BatchNorm1d(96),
			nn.LeakyReLU(),
			nn.MaxPool2d(2))
		self.fc = nn.Sequential(
            nn.Linear(48*45, 32))
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		#print("out size: ", out.size())
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out

class ClassificationNet(nn.Module):
	def __init__(self, input_size, num_classes):
		super(ClassificationNet, self).__init__()
		self.fc1 = nn.Linear(input_size, 1024)
		self.fc2 = nn.Linear(1024, 256)
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
        self.subjects = []
        pkl_file = open(dataset_path, 'rb')
        self.subjects = pickle.load(pkl_file)
        pkl_file.close()
        # i = 0
        # for subject in dataset:
            # time_series = subject[0]
            # allCombinations = []
            # num_of_net = time_series.shape[0]
            # for net1_index in range(num_of_net):
                # for net2_index in range(net1_index+1,num_of_net):
                    # two_nets = np.concatenate((time_series[net1_index],time_series[net2_index]))
                    # allCombinations.append(two_nets)
            # subject = (np.asarray(allCombinations,dtype=np.float32),subject[1])
            # self.subjects.append(subject)
            # print(i)
            # i+=1
				
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        time_series, score = self.subjects[idx]
        return torch.from_numpy(time_series), score
	
#This function enable the model to run in cpu and gpu	
def to_cuda(x):
    use_gpu = torch.cuda.is_available()	
    device = torch.device("cuda:1")
    if use_gpu:
        x = x.to(device)
    return x
	
