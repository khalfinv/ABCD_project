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
        self.similarity_measure = SimilarityMeasureNetwork(750,1)
        self.similarity_measure = to_cuda(self.similarity_measure)
        self.classification_net = ClassificationNet(9045,3)
        self.classification_net = to_cuda(self.classification_net)
		
		
    def forward(self, x):
        #print ("x:", x.shape)
        subjects = to_cuda(torch.zeros([x.shape[0],9045], dtype=torch.float32, requires_grad=True))
        subject_idx = 0
        for subject in x:                     
            fc_all=self.similarity_measure(subject)
            subjects[subject_idx] = fc_all.squeeze(1)
            subject_idx+=1
        #print("subjects:",subjects.cpu())
        out_class = self.classification_net(subjects)
        return out_class
		

class ClassificationNet(nn.Module):
	def __init__(self, input_size, num_classes):
		super(ClassificationNet, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1)	
		self.fc1 = nn.Sequential(
			nn.Linear(input_size, 1024),
			nn.Dropout(p=0.3),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(1024, 256),
			nn.Dropout(p=0.3),
			nn.ReLU())
		self.fc3 = nn.Sequential(
			nn.Linear(256, 64),
			nn.Dropout(p=0.3),
			nn.ReLU())
		self.fc4 = nn.Sequential(
			nn.Linear(64, num_classes))


	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		return self.logsoftmax(out)
		

		
class SimilarityMeasureNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimilarityMeasureNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Dropout(p=0.3),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Dropout(p=0.3),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.Dropout(p=0.3),
            nn.ReLU())
        self.fc4 = nn.Sequential(
            nn.Linear(8, num_classes),
            nn.Tanh())
			
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        # count = 0
        # for i in out:
            # for j in i:
                # if j == 0:
                    # count+=1
        # print ("num of zeros:", count)
        out = self.fc4(out)
        print("SimilarityMeasureNetwork out", out)
        return out
		

		
class TimeseriesDataset(data.Dataset):
    def __init__(self, dataset_path):
        self.subjects = []
        pkl_file = open(dataset_path, 'rb')
        self.subjects = pickle.load(pkl_file)
        pkl_file.close()
				
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        time_series, score = self.subjects[idx]
        allCombinations = []
        num_of_net = time_series.shape[0]
        for net1_index in range(num_of_net):
            for net2_index in range(net1_index+1,num_of_net):
                two_nets = np.concatenate((time_series[net1_index],time_series[net2_index]))
                allCombinations.append(two_nets)
        new_time_series = np.asarray(allCombinations,dtype=np.float32)
        return torch.from_numpy(new_time_series), score
	
#This function enable the model to run in cpu and gpu	
def to_cuda(x):
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0")
    if use_gpu:
        x = x.to(device)
    return x
	
