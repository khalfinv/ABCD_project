import torch	
import argparse
import common
import os
		   
def load_checkpoint(model,filepath):
    # "lambda" allows to load the model on cpu in case it is saved on gpu
    state = torch.load(filepath,lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    return model

	
def testFunc(net, test_loader):
	lossSum = 0 # sum of all loss 
	errSum = 0 # sum of all error rates 
	total = 0 # sum of total scores 
	net.eval() # turning the network to evaluation mode, affect dropout and batch-norm layers if exists
	for i, (time_series, scores) in enumerate(validate_loader):
		time_series = time_series.unsqueeze(1)
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
	parser.add_argument('--model', required=True, help='path to the model after training')
	args = parser.parse_args()
	
	test_dataset = common.TimeseriesDataset(os.path.join(args.data_folder,"test_set_class.pkl"))
	
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											   shuffle=False)
	net = common.fMRI_CNN()
	#load the model						   
	net = load_checkpoint(net, args.model)
	net = common.to_cuda(net)
	# Test the Model
	testErr = testFunc(net, test_loader)
	print (testErr)
