from __future__ import division
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

np.set_printoptions(threshold=np.nan)

def sigmoid(x):
	return np.reciprocal(1 + np.exp(-x))

'''
Trains a set of n models to learn each of the dataset labels and returns n sets of learned weights.
'''
def fit_model(X, y, w = None, learning_rate = 0.25, threshold_value = 0.01, reg_coef = 1):
	unique_labels = np.unique(y)
	# Create initial weights for each model to train
	if w is None:
		print "Providing default weights of ones"
		w = np.ones((X.shape[1] + 1, unique_labels.shape[0]))

	X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

	num_samples = X.shape[0]
	
	for label_idx in range(unique_labels.shape[0]):
		print "Now building model for label ", unique_labels[label_idx], "..."
		# print "Training model for label: ", unique_labels[label_idx]
		# Set labels to {0,1}; 1 = current model's label, 0 = all other labels 
		y_curr_model = np.array([y == unique_labels[label_idx]]).astype(int)
		current_weights = w[:, label_idx]

		while True:
			current_model_errors = (sigmoid(np.dot(X,current_weights)) - y_curr_model).transpose()[:,0]
			updated_weights = current_weights - (learning_rate/num_samples) * np.dot(X.transpose(),current_model_errors)
			updated_weights[1:] -= (learning_rate*reg_coef/num_samples) * current_weights[1:] # Regularize non-bias weights (L2-norm)
			if np.all( ((np.absolute(current_weights - updated_weights)) < threshold_value)):
				print "Finished building model for label ", unique_labels[label_idx]
				print "Current weights: ", current_weights
				print "Updated (final) weights: ", updated_weights
				w[:, label_idx] = updated_weights	# Set weights for the current model and continue to next model
				break
			else:
				current_weights = updated_weights
	return w

'''
Make prediction based on the model that returns the maximum probability, i.e. max_i(P(y=i)|x,w).
'''
def model_predict(X, y, w):
	X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
	return np.argmax(sigmoid(X.dot(w)), axis=1)

digits_dataset = datasets.load_digits()
X,y = digits_dataset.data, digits_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

print "Training set labels (1-50): ", y_train[0:50]

opt_weights = fit_model(X_train, y_train)

predictions = model_predict(X_test, y_test, opt_weights)

correct_preds_bool_arr = (predictions == y_test)

log_reg_cfr = LogisticRegression()
log_reg_cfr.fit(X_train, y_train)

print "Custom model intercepts: ", opt_weights[0,:]
print "Sklearn model intercepts: ", log_reg_cfr.intercept_

print "Custom model coef: ", opt_weights[1:,:]
print "Sklearn model coef: ", log_reg_cfr.coef_

print "Custom model accuracy on digits dataset (multiclass dataset): {}".format(np.count_nonzero(correct_preds_bool_arr)/predictions.shape[0])
print "Sklearn model accuracy on digits dataset (multiclass dataset): {}".format(log_reg_cfr.score(X_test,y_test))