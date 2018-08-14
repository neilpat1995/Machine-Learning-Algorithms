from __future__ import division
import numpy as np
from collections import Counter
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split

class TreeNode(object):
	def __init__(self, attribute = None, label = None):
		self.attribute = attribute
		self.label = label
		self.children = []
		self.children_attr_values = []

class DecisionTree(object):
	def __init__(self, split_method=None):
		if split_method is None:
			self.split_method = 'entropy'
		else:
			self.split_method = split_method
		root = None

'''
Helper function to compute entropy of dataset 'S'; the label is assumed to be the last column of S. 
'''
def calculate_entropy(S):
	dataset_label_values = S[:,-1]
	entropy = 0
	counter = Counter(dataset_label_values)
	for label_class in counter.keys():
		prob_class = counter[label_class] / len(dataset_label_values)
		entropy -= (prob_class * math.log(prob_class, 2))
	return entropy

'''
Main function to construct decision tree to classify samples in dataset 'S'. 
The label is assumed to be the last column in S, and the dataset must contain the attributes specified by 'attribute_labels'.
'''
def construct_tree(S, attribute_labels):
	# Base case: entropy = 0, i.e. no uncertainty in label for current dataset. Construct leaf node for this label.
	if calculate_entropy(S) == 0:
		return TreeNode(label = S[0,-1])

	# At least 1 attribute left to split on; calculate information gain for each attribute to determine optimal split.
	info_gains = []
	for attribute_idx in range(len(attribute_labels) - 1): # Exclude label name
		info_gain = calculate_entropy(S)
		counter = Counter(S[:,attribute_idx])
		for split_value in counter.keys():	# Compute information gain using each value of this attribute
			prob_split = counter[split_value] / S.shape[0]
			split_entropy = calculate_entropy(S[S[:,attribute_idx] == split_value, :])
			info_gain -= prob_split * split_entropy
		info_gains.append(info_gain)
	optimal_attribute_idx = info_gains.index(max(info_gains))
	print 'Determined optimal attribute as: ', attribute_labels[optimal_attribute_idx]


	# Perform recursive calls using the determined optimal splitting attribute
	treeNode = TreeNode(attribute = attribute_labels[optimal_attribute_idx])

	optimal_attribute_values = Counter(S[:, optimal_attribute_idx])
	for optimal_attribute_value in optimal_attribute_values.keys():
		treeNode.children_attr_values.append(optimal_attribute_value)
		# Create sub-dataset for recursive call to construct_tree() and remove the current node's attribute
		split_dataset = S[S[:,optimal_attribute_idx] == optimal_attribute_value, :]
		split_dataset = np.delete(split_dataset, optimal_attribute_idx, axis = 1)
		split_attribute_labels = attribute_labels[:optimal_attribute_idx] + attribute_labels[optimal_attribute_idx + 1:]
		print 'Now recursing with dataset: '
		print split_dataset
		print 'And with remaining attribute labels: '
		print split_attribute_labels
		print "On attribute value: ", optimal_attribute_value
		treeNode.children.append(construct_tree(split_dataset, split_attribute_labels))
	return treeNode

'''
Prediction function that classifies samples in test set 'S' using a decision tree model rooted at 'root'.
'S' must be of shape [n_samples, n_features], where features are in the order specified by 'attribute_labels'. 
'''
def predict(S, attribute_labels, root):
	predictions = np.empty([S.shape[0]], dtype=object)
	for sample_idx in range(S.shape[0]):
		treeNode = root
		while treeNode.label is None:
			split_attribute = treeNode.attribute
			attribute_index = attribute_labels.index(split_attribute)
			sample_attribute_value = S[sample_idx, attribute_index]
			treeNode = treeNode.children[treeNode.children_attr_values.index(sample_attribute_value)]
		predictions[sample_idx] = treeNode.label
	return predictions

if __name__ == '__main__':
	# Test using small sample dataset 
	S = np.array([["sunny", "hot", "high", "false", "no"],
		["sunny", "hot", "high", "true", "no"],
		["overcast", "hot", "high", "false", "yes"],
		["rainy", "mild", "high", "false", "yes"],
		["rainy", "cool", "normal", "false", "yes"],
		["rainy", "cool", "normal", "true", "no"],
		["overcast", "cool", "normal", "true", "yes"],
		["sunny", "mild", "high", "false", "no"],
		["sunny", "cool", "normal", "false", "yes"],
		["rainy", "mild", "normal", "false", "yes"],
		["sunny", "mild", "normal", "true", "yes"],
		["overcast", "mild", "high", "true", "yes"],
		["overcast", "hot", "normal", "false", "yes"],
		["rainy", "mild", "high", "true", "no"]
	])
	S_attribute_labels = ["outlook", "temp", "humidity", "windy", "play"]

	# Split into train/testing sets and train/test accordingly
	S_train, S_test = S[:10, :], S[10:, :]
	model_tree = construct_tree(S_train, S_attribute_labels)
	print 'Predictions on samples 10-14: ', predict(S_test, S_attribute_labels, model_tree)