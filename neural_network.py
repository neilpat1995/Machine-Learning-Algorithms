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

	# print "length of w: ", len(w)
	# print "w[0].shape: ", w[0].shape
	# print "w[1].shape: ", w[1].shape

	iteration = 1

	while True:
		# Intialize parameters
		upper_deltas = []
		D = []
		for weight_matrix_idx in range(len(w)):
			upper_deltas.append(np.zeros((w[weight_matrix_idx].shape[0], w[weight_matrix_idx].shape[1])))
			D.append(np.zeros((w[weight_matrix_idx].shape[0], w[weight_matrix_idx].shape[1])))

		# print "Upper deltas shape: "
		# for i in upper_deltas:
		# 	print i.shape
		# print "Upper deltas length: ", len(upper_deltas)
		# print "D shape: "
		# for j in D:
		# 	print j.shape
		# print "D length: ", len(D)

		# break


		for sample_idx in range(X.shape[0]):
			activations = forward_propagation(X[sample_idx,:], w)
			# print "activations.shape: "
			# for i in activations:
			# 	print i.shape
			deltas = [[None]] * (len(activations) - 1) # Exclude computation of errors for input layer
			deltas[-1] = activations[-1] - y[sample_idx, :]	# Output error
			# Compute deltas for all hidden layers
			for layer_idx in range(len(deltas) - 2, -1, -1):
				deltas[layer_idx] = np.matmul(np.transpose(w[layer_idx + 1]), deltas[layer_idx + 1])
				deltas[layer_idx] = np.multiply(deltas[layer_idx], activations[layer_idx + 1])
				deltas[layer_idx] = np.multiply(deltas[layer_idx], 1 - activations[layer_idx + 1])
				deltas[layer_idx] = deltas[layer_idx][1:] # Remove bias unit error

			# print "deltas.shape: "
			# for i in deltas:
			# 	print i.shape
			# break

			for l in range(len(upper_deltas)):
				update_lower_delta = np.expand_dims(deltas[l], axis=1)
				update_activation = np.expand_dims(activations[l], axis=0)
				# print "l: ", l
				# print "1st param: ", update_lower_delta.shape
				# print "2nd param: ", update_activation.shape
				# print "upper_deltas[l].shape: ", upper_deltas[l].shape
				upper_deltas[l] += np.matmul(update_lower_delta, update_activation)


		for l in range(len(D)):
			D[l] = (1. / X.shape[0]) * upper_deltas[l]
			D[l][:, 1:] += reg_coef * w[l][:, 1:] # Add regularization term for non-bias terms (exclude first column)

		w_updated = [ [] ] * len(w)
		threshold_exceeded = False

		for l in range(len(w)):
			w_updated[l] = w[l] - learning_rate * D[l]
			if not np.all( ((np.absolute(w[l] - w_updated[l])) < threshold_value)):
				threshold_exceeded = True

		if not threshold_exceeded:
			return w_updated	# Optimal parameters found

		w = w_updated	# Continue gradient descent process with updated parameter set
		if iteration == 1:
			print "updated weights: ", w
		iteration += 1


if __name__ == '__main__':
	# opt_params = fit_model(np.ones((7,5)), np.ones((7,4)), (3,))
	# print "----------------"
	# print "opt params: ", opt_params

	dataset = load_iris()
	X = dataset.data
	y = dataset.target

	# Binarize labels for model fitting and prediction
	unique_labels = np.unique(y)
	lb = preprocessing.LabelBinarizer()
	lb.fit(unique_labels)
	y_transform = lb.transform(y)

	# print "y_transform.shape: ", y_transform.shape
	# print "y_transform: ", y_transform
	
	X_train, X_test, y_train, y_test = train_test_split(X, y_transform, test_size = 0.98) # Train on 3 samples for gradient checking
	print "X_train: ", X_train
	print "y_train: ", y_train

	optimal_weights = fit_model(X_train, y_train, (5,))
	print "Optimal weights: ", optimal_weights