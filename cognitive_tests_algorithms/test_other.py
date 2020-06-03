
#!/usr/bin/python3
import os, sys, argparse, pickle
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import preprocessing
import sklearn.metrics as skm
import numpy as np
from sklearn import svm

def load_data(pkl_path):
	pkl_file = open(pkl_path, 'rb')
	subjects = pickle.load(pkl_file)
	pkl_file.close()
	X = subjects["X"]
	y = subjects["y"]	
	return X, y



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_folder', required=True, help='path to folder contaning all the data - train, validate and test')
	args = parser.parse_args()
			
			
	train_path = os.path.join(args.data_folder,"train_set.pkl")
	test_path = os.path.join(args.data_folder,"test_set.pkl")
	
	x_train, y_train = 	load_data(train_path)
	#std_scale = preprocessing.StandardScaler().fit(x_train)
	#x_train_norm = std_scale.transform(x_train)
	#print("mean: ",std_scale.mean_.tolist(), "std: ", std_scale.var_.tolist()) 
	x_test, y_test = 	load_data(test_path)
	#weigth_dict = {0: 0.68, 1: 0.22, 2: 1.49}
	#weigth_dict = {0: 0.2, 2: 0.8}
	print("Train statistics:")
	values, counts = np.unique(y_train, return_counts=True)
	percentage = counts / len(x_train)
	print("The classes: " + str(values) + " The frequency: " + str(counts) + " The percentage: " + str(percentage) + "\n")
	
	print("Test statistics:")
	values, counts = np.unique(y_test, return_counts=True)
	percentage = counts / len(x_test)
	print("The classes: " + str(values) + " The frequency: " + str(counts) + " The percentage: " + str(percentage) + "\n")
	
	print("***** Logistic Regression*****")
	#clf_log = LogisticRegression(random_state=0,class_weight = weigth_dict).fit(x_train_norm, y_train)
	clf_log = LogisticRegression(random_state=0, max_iter = 300).fit(x_train, y_train)
	# print(clf_log.coef_)
	#x_test_norm = std_scale.transform(x_test)
	test_prediction = clf_log.predict(x_test)
	test_acc = clf_log.score(x_test, y_test)
	train_prediction = clf_log.predict(x_train)
	train_acc = clf_log.score(x_train, y_train)
	print( "Test report: " )
	print(skm.classification_report(y_test, test_prediction))
	print( "Train report: ")
	print(skm.classification_report(y_train, train_prediction))
	
	print("******** Decision Tree *******")
	clf_tree = tree.DecisionTreeClassifier()
	clf_tree = clf_tree.fit(x_train, y_train)
	test_prediction = clf_tree.predict(x_test)
	test_acc = clf_tree.score(x_test, y_test)
	train_prediction = clf_tree.predict(x_train)
	train_acc = clf_tree.score(x_train, y_train)
	print( "Test report: " )
	print(skm.classification_report(y_test, test_prediction))
	
	print("******** SVM *******")
	clf_svm = svm.SVC(kernel='sigmoid')
	clf_svm = clf_svm.fit(x_train, y_train)
	test_prediction = clf_svm.predict(x_test)
	test_acc = clf_svm.score(x_test, y_test)
	train_prediction = clf_svm.predict(x_train)
	train_acc = clf_svm.score(x_train, y_train)
	print(skm.confusion_matrix(y_test, test_prediction))
	print( "Test report: " )
	print(skm.classification_report(y_test, test_prediction))
	print( "Train report: ")
	print(skm.classification_report(y_train, train_prediction))
	
	
