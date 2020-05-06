
import sys
sys.path.insert(1, 'fmri_cnn')
import common

train_dataset = common.TimeseriesDataset("train_set_class.pkl")
len(train_dataset)
train_dataset[10]