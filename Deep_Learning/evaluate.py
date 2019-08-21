import torch	
import argparse
import common
import os
from sklearn.metrics import r2_score
		   
def load_checkpoint(model,filepath):
    # "lambda" allows to load the model on cpu in case it is saved on gpu
    state = torch.load(filepath,lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    return model

	
def testFunc(cnn, test_loader):
	cnn.eval() # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
	outputs_all = torch.zeros(0)
	scores_all = torch.zeros(0).float()
	for i, (time_series, scores) in enumerate(test_loader):
		time_series = time_series.unsqueeze(1)
		time_series = common.to_cuda(time_series)
		outputs = cnn(time_series)
		outputs_all = torch.cat([outputs_all, outputs.reshape(-1)])
		scores_all = torch.cat([scores_all, scores.float()])
	r_squared = r2_score(scores_all.tolist(),outputs_all.tolist()) 
	return r_squared * 100
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_folder', required=True, help='path to folder contaning all the data - train, validate and test')
	parser.add_argument('--model', required=True, help='path to the model after training')
	args = parser.parse_args()
	
	test_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"test_set.pkl"))
	
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											   shuffle=False)
	net = common.FeatureExtractor()
	#load the model						   
	net = load_checkpoint(net, args.model)
	net = common.to_cuda(net)
	# Test the Model
	testErr = testFunc(net, test_loader)
	print (testErr)
