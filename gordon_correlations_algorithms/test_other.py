
#!/usr/bin/python3
import os, sys, argparse, pickle
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import preprocessing
import numpy as np

def load_data(pkl_path):
	pkl_file = open(pkl_path, 'rb')
	subjects = pickle.load(pkl_file)
	pkl_file.close()
	res = list(zip(*subjects)) 
	x = res[0]
	y = res[1]
	return x, y



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_folder', required=True, help='path to folder contaning all the data - train, validate and test')
	args = parser.parse_args()
			
			
	train_path = os.path.join(args.data_folder,"train_set_class.pkl")
	test_path = os.path.join(args.data_folder,"test_set_class.pkl")
	
	x_train, y_train = 	load_data(train_path)
	std_scale = preprocessing.StandardScaler().fit(x_train)
	x_train_norm = std_scale.transform(x_train)
	print("mean: ",std_scale.mean_.tolist(), "std: ", std_scale.var_.tolist()) 
	x_test, y_test = 	load_data(test_path)
	#weigth_dict = {0: 0.68, 1: 0.22, 2: 1.49}
	weigth_dict = {0: 1, 1: 1, 2: 1}
	print("***** Logistic Regression*****")
	clf_log = LogisticRegression(random_state=0,class_weight = weigth_dict).fit(x_train_norm, y_train)
	x_test_norm = std_scale.transform(x_test)
	test_prediction = clf_log.predict(x_test_norm)
	test_acc = clf_log.score(x_test_norm, y_test)
	print("TEST: number of below average:", np.count_nonzero(test_prediction == 0), " number of average:", np.count_nonzero(test_prediction == 1), " number of above average:", np.count_nonzero(test_prediction == 2))
	print("test accuracy:", test_acc)
	train_prediction = clf_log.predict(x_train)
	train_acc = clf_log.score(x_train, y_train)
	print("TRAIN: number of below average:", np.count_nonzero(train_prediction == 0), " number of average:", np.count_nonzero(train_prediction == 1), " number of above average:", np.count_nonzero(train_prediction == 2))	
	print("train_accuracy:", train_acc)
	
	print("******** Decision Tree *******")
	clf_tree = tree.DecisionTreeClassifier()
	clf_tree = clf_tree.fit(x_train, y_train)
	test_prediction = clf_tree.predict(x_test)
	test_acc = clf_tree.score(x_test, y_test)
	print("TEST: number of below average:", np.count_nonzero(test_prediction == 0), " number of average:", np.count_nonzero(test_prediction == 1), " number of above average:", np.count_nonzero(test_prediction == 2))
	print("test accuracy:", test_acc)
	train_prediction = clf_tree.predict(x_train)
	train_acc = clf_tree.score(x_train, y_train)
	print("TRAIN: number of below average:", np.count_nonzero(train_prediction == 0), " number of average:", np.count_nonzero(train_prediction == 1), " number of above average:", np.count_nonzero(train_prediction == 2))	
	print("train_accuracy:", train_acc)