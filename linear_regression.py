from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


diabetes = datasets.load_diabetes()
print "(Data shape) (Labels shape) = ", diabetes.data.shape, diabetes.target.shape
print "Label distribution: "
unique_labels, label_freqs = np.unique(diabetes.target, return_counts=True)
for label_idx in range(unique_labels.size):
	if label_freqs[label_idx] != 0:
		print unique_labels[label_idx], ": ", label_freqs[label_idx]

'''
TODO: Encapsulate below lin. reg. functions into a class 
class SimpleLinearRegression:
	def __init__(self, )

'''

def fit_model(X, y, w = None, learning_rate = 0.25, threshold_value = 0.001):
	ones_vector = np.ones((X.shape[0],1))	# Prepend samples with bias feature of ones
	X = np.concatenate( (ones_vector, X), axis=1)
	if w is None:
		w = np.ones( (X.shape[1]) )	# Initialize default weights to ones
	n_samples = X.shape[0]
	current_weights = w
	# Update weights until all weights converge
	while True:
		updated_weights = np.zeros(w.shape[0])
		for w_i in range(w.shape[0]):
			updated_weights[w_i] = current_weights[w_i] - (learning_rate / n_samples) * np.sum( np.multiply((X.dot(current_weights) - y), X[:,w_i])) 
		if np.all( ((np.absolute(current_weights - updated_weights)) < threshold_value)):
			return updated_weights
		else:
			current_weights = updated_weights


'''
Prediction function that returns both the mean squared error over the test sample space, individual
predictions for each sample, and rounded predictions for each sample to match integer label values.
'''
def model_predict(X, y, w):
	ones_vector = np.ones((X.shape[0],1))
	X = np.concatenate( (ones_vector, X), axis=1)
	num_samples = X.shape[0]
	return (np.sum(np.square(( X.dot(w) - y ))) / num_samples, X.dot(w), np.around(X.dot(w)))

'''
TESTING CUSTOM LinearRegression MODEL
'''

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size = 0.2)

custom_weights = fit_model(X_train, y_train)
test_acc, test_preds, rounded_preds = model_predict(X_test, y_test, custom_weights)
print 'Custom linear regression model prediction MSE = ', test_acc
print 'Custom linear regression model predictions: ', test_preds

# print "Custom model accuracy with rounded prediction labels: {}".format(np.count_nonzero(rounded_preds == y_test)/rounded_preds.size)


'''
TESTING SKLEARN LinearRegression MODEL
'''

reg_clf = LinearRegression()
reg_clf.fit(X_train, y_train)

sklearn_preds = reg_clf.predict(X_test)
print 'Sklearn linear regression model prediction MSE = ', np.sum(np.square( sklearn_preds - y_test )) / X_test.shape[0]
print 'Sklearn linear regression model predictions: ', sklearn_preds
# print 'Sklearn linear regression model rounded prediction accuracy: ', np.count_nonzero(np.around(sklearn_preds) == y_test)/sklearn_preds.size