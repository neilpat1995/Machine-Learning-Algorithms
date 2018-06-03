from __future__ import division
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

def sigmoid(x):
	return np.reciprocal(1 + np.exp(-x))
	#return  np.array([1]) / np.array(1 + math.exp(-x))

def fit_model(X, y, w = None, learning_rate = 0.25, threshold_value = 0.001):
	if w is None:
		w = np.ones(X.shape[1] + 1)
	X = np.concatenate((np.ones(X.shape[0]).reshape(X.shape[0],1), X), axis=1)

	current_weights = w
	num_samples = X.shape[0]
	
	while True:
		updated_weights = current_weights - (learning_rate/num_samples) * np.dot(X.transpose(),(sigmoid(np.dot(X,current_weights)) - y))
		if np.all( ((current_weights - updated_weights) < threshold_value)):
			return updated_weights
		else:
			current_weights = updated_weights

def model_predict(X, y, w):
	X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
	return sigmoid(X.dot(w))


digits_dataset = datasets.load_digits(n_class=2)
X,y = digits_dataset.data, digits_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

opt_weights = fit_model(X_train, y_train)
predictions = model_predict(X_test, y_test, opt_weights)

predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0

correct_preds_bool_arr = (predictions == y_test)

print "Custom model accuracy on digits dataset with 2 classes: {}".format(np.count_nonzero(correct_preds_bool_arr)/predictions.shape[0])