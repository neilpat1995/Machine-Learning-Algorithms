from __future__ import division
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

class NeuralNetwork():

	def __init__(self):
		return self




'''
Sklearn MLP Classifer
'''

# mlp_clf = MLPClassifier()

def sigmoid(x):
	return np.reciprocal(1 + np.exp(-x))

def forward_propagation(x, w):
	# print "x: ", x, ", type(x): ", type(x)
	# print "w: ", w
	activations = [x]
	for weight_matrix_idx in range(len(w)):
		activations.extend([np.zeros(w[weight_matrix_idx].shape[0])]) # List of numpy arrays representing activations (for all layers)
	# print "activations: ", activations

	for layer_idx in range(len(w)):
		# print "activations[layer_idx].shape: ", activations[layer_idx].shape
		activations[layer_idx] = np.concatenate(([1], activations[layer_idx]))
		activations[layer_idx + 1] = sigmoid(w[layer_idx].dot(activations[layer_idx]))

	return activations

'''
Training algorithm that learns optimal network weights using backpropagation.
'''
def fit_model(X, y, hidden_layer_sizes, w=None, learning_rate=0.25, threshold_value=0.01, reg_coef=1):
	'''
	Parameters:
	hidden_layer_sizes: a tuple indicating the sizes of all hidden layers in order. Excludes input and output layer sizes.
		-> Future step: Remove this and instead read in from a class field initialized in a constructor.
	w: a list of (N - 1) weight matrices for each network layer, where N = number of total network layers
	'''

	if w is None:
		w = []
		num_network_nodes = [X.shape[1]]
		num_network_nodes.extend(list(hidden_layer_sizes))
		num_network_nodes.append(y.shape[1]) # List of number of nodes per layer excluding all biases
		
		for layer_idx in range(len(num_network_nodes) - 1):
			w.append(np.ones((num_network_nodes[layer_idx + 1], num_network_nodes[layer_idx] + 1))) # Use initial weights of ones

	while True:
		# Intialize parameters
		upper_deltas = []
		D = []
		for weight_matrix_idx in range(len(w)):
			upper_deltas.extend([np.zeros((w[weight_matrix_idx].shape[0], w[weight_matrix_idx].shape[1]))])
			D.extend([np.zeros((w[weight_matrix_idx].shape[0], w[weight_matrix_idx].shape[1]))])


		for sample_idx in range(X.shape[0]):
			activations = forward_propagation(X[sample_idx,:], w)
			deltas = [None] * (len(activations) - 1)
			deltas[len(deltas) - 1] = activations[len(activations) - 1] - y[sample_idx, :]	# Output error
			# Compute deltas for all hidden layers
			for layer_idx in range(len(deltas) - 2, -1, -1):
				deltas[layer_idx] = np.matmul(np.transpose(w[layer_idx + 1]), deltas[layer_idx + 1])
				deltas[layer_idx] = np.multiply(deltas[layer_idx], activations[layer_idx + 2])
				deltas[layer_idx] = np.multiply(deltas[layer_idx], 1 - activations[layer_idx + 2])

			for l in range(len(upper_deltas)):
				print "l: ", l
				print "upper deltas[l].shape: ", upper_deltas[l].shape
				print "deltas[l+1].shape: ", deltas[l].shape
				print "np.transpose(activations[l]).shape: ", np.transpose(activations[l]).shape
				upper_deltas[l] += np.matmul(deltas[l], np.transpose(activations[l]))

		for i in range(len(D)):
			D[i] = (1. / X.shape[0]) * upper_deltas[i]
		break




if __name__ == '__main__':
	fit_model(np.ones((7,5)), np.ones((7,4)), (3,))