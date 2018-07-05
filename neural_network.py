from __future__ import division
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

class NeuralNetwork():

	def __init__(self):
		return self


EPSILON = 3

def sigmoid(x):
	return np.reciprocal(1 + np.exp(-x))

def forward_propagation(x, w):
	# print "x: ", x, ", type(x): ", type(x)
	# print "w: ", w
	activations = [x]
	# print 'activations (x) shape: ', activations[0].shape
	for weight_matrix_idx in range(len(w)):
		activations.append(np.zeros(w[weight_matrix_idx].shape[0])) # List of numpy arrays representing activations (for all layers)
	# print "activations: ", activations

	for layer_idx in range(len(w)):
		# print "activations[layer_idx].shape: ", activations[layer_idx].shape
		# print "layer_idx: ", layer_idx
		# print "activations[layer_idx]: ", activations[layer_idx]
		# print "activations[layer_idx].shape: ", activations[layer_idx].shape
		# if activations[layer_idx].ndim == 1:
		# 	activations[layer_idx] = np.expand_dims(activations[layer_idx], axis=1)
		activations[layer_idx] = np.concatenate(([1], activations[layer_idx]))
		activations[layer_idx + 1] = sigmoid(w[layer_idx].dot(activations[layer_idx]))

	return activations

'''
Training algorithm that learns optimal network weights using backpropagation.
'''
def fit_model(X, y, hidden_layer_size, w=None, learning_rate=0.25, threshold_value=0.001, reg_coef=0.001):
	'''
	Parameters:
	hidden_layer_size: an integer indicating the number of hidden layer activation units.
		-> Future step: Support arbitrary numbers of hidden layers. Remove this and instead read in from a 
		class field initialized in a constructor.
	w: a list of 2 weight matrices for each network layer pair, mapping from input layer-> hidden layer and hidden layer-> output layer
	'''

	# Handle single-sample inputs, i.e. one-dimensional training data and labels.

	if X.ndim == 1:
		X = np.expand_dims(X, axis=0)

	if y.ndim == 1:
		y = np.expand_dims(y, axis=0)

	if w is None:
		w = []
		num_network_nodes = [X.shape[1], hidden_layer_size, y.shape[1]] # List of number of nodes per layer excluding all biases
		for layer_idx in range(len(num_network_nodes) - 1):
			# Randomly initialize weights between [-EPSILON, EPSILON]
			w.append(np.random.rand(num_network_nodes[layer_idx + 1], num_network_nodes[layer_idx] + 1) * (2 * EPSILON) - EPSILON)

	while True:
		# Intialize parameters
		upper_deltas = []
		D = []
		for weight_matrix_idx in range(len(w)):
			upper_deltas.append(np.zeros((w[weight_matrix_idx].shape[0], w[weight_matrix_idx].shape[1])))
			D.append(np.zeros((w[weight_matrix_idx].shape[0], w[weight_matrix_idx].shape[1])))

		for sample_idx in range(X.shape[0]):
			activations = forward_propagation(X[sample_idx,:], w)
			deltas = [[None]] * (len(activations) - 1) # Exclude computation of errors for input layer
			deltas[-1] = activations[-1] - y[sample_idx, :]	# Output error
			# Compute deltas for all hidden layers
			for layer_idx in range(len(deltas) - 2, -1, -1):
				deltas[layer_idx] = np.matmul(np.transpose(w[layer_idx + 1]), deltas[layer_idx + 1])
				deltas[layer_idx] = deltas[layer_idx] * activations[layer_idx + 1]
				deltas[layer_idx] = deltas[layer_idx] * (1 - activations[layer_idx + 1])
				deltas[layer_idx] = deltas[layer_idx][1:] # Remove bias unit error

			for l in range(len(upper_deltas)):
				update_lower_delta = np.expand_dims(deltas[l], axis=1)
				update_activation = np.expand_dims(activations[l], axis=0)
				upper_deltas[l] += np.matmul(update_lower_delta, update_activation)

		for l in range(len(D)):
			D[l] = (1. / X.shape[0]) * upper_deltas[l]
			D[l][:, 1:] += reg_coef * w[l][:, 1:] # Add regularization term for non-bias terms (exclude first column)

		w_updated = [ [] ] * len(w)
		threshold_exceeded = False

		for l in range(len(w)):
			w_updated[l] = w[l] - learning_rate * D[l]
			if not np.all(np.absolute(w[l] - w_updated[l]) < threshold_value):
				threshold_exceeded = True

		if not threshold_exceeded:
			return w_updated	# Optimal parameters found

		w = w_updated	# Continue gradient descent process with updated parameter set

if __name__ == '__main__':

	dataset = load_iris()
	X = dataset.data
	y = dataset.target

	# Transform labels into binary vectors for model fitting and prediction
	unique_labels = np.unique(y)
	lb = preprocessing.LabelBinarizer()
	lb.fit(unique_labels)
	y_transform = lb.transform(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y_transform, test_size = 0.25)

	print "Training model..."
	optimal_weights = fit_model(X_train, y_train, 100)
	print "Optimal weights: ", optimal_weights


	test_predictions_transform = [[]] * X_test.shape[0]
	
	print "Testing model..."
	for i in range(len(test_predictions_transform)):
		test_predictions = forward_propagation(X_test[i], optimal_weights) # Last list holds activations using the optimal weights, i.e. the labels
		test_predictions = test_predictions[-1]
		test_predictions_transform[i] = (test_predictions == np.amax(test_predictions)).astype(int) # Predicts label with the highest activation

	print "Computing accuracy..."
	n_correct_preds = 0
	for test_sample_idx in range(len(y_test)):
		if np.array_equal(test_predictions_transform[test_sample_idx], y_test[test_sample_idx]):
			n_correct_preds += 1

	print "Test accuracy: ", (n_correct_preds / len(y_test))


	# mlp = MLPClassifier(activation='logistic', learning_rate_init=0.25)
	# mlp.fit(X_train, y_train)
	# mlp_preds = mlp.predict(X_test)
	# print "mlp_preds.shape: ", mlp_preds.shape
	# for prediction in range(len(mlp_preds)):
	# 	print "Prediction before binarizing: ", mlp_preds[prediction]
	# 	mlp_preds[prediction] = (mlp_preds[prediction] == np.amax(mlp_preds[prediction])).astype(int)

	# print "mlp predictions after binarizing: ", mlp_preds

	# mlp_n_correct_preds = 0
	# for test_sample_idx in range(len(mlp_preds)):
	# 	# print "test_predictions_transform[", test_sample_idx, "]: ", test_predictions_transform[test_sample_idx]
	# 	# print "y_test[", test_sample_idx, "]: ", y_test[test_sample_idx]
	# 	if np.array_equal(mlp_preds[test_sample_idx], y_test[test_sample_idx]):
	# 		n_correct_preds += 1

	# print "Sklearn test accuracy: ", (mlp_n_correct_preds / len(y_test))