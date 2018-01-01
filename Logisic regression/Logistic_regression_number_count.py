"""
Logistic regression to teach the machine how to count 
X = 1 -> y = 2
X = 5 -> y = 6
X = 9 -> y = 0
using python 2.7 / Numpy and matplotlib for visualization 
"""

import numpy as np
import matplotlib as plt

train_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 5]
test_x = []
train_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 6]
test_y = []

def transform_to_binary(X, y, dim, classes):
	m = len(X)
	bin_X = np.zeros((dim, m))
	bin_y = np.zeros((m, classes))
	# Tranform X to binary 1 of 4(dim) bits -> 0001
	for i in range(0, m):
		string = str(np.binary_repr(X[i], width=dim))
		for j in range (0, dim):
			bin_X[j, i] = string[j]

	# Transform y to binary 1 -> 0100000000
	temp = np.zeros((classes, ))
	for i in range(0, m):
		temp[y[i]] = 1
		bin_y[i, :] = temp
		temp[y[i]] = 0
	return bin_X, bin_y

# Helper function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# Initialize our parameters (weights W and bias b with zeros) 
def initialize(dim, classes):
	w = None
	b = None
	w = np.zeros((dim, classes))
	b = np.zeros((1, classes))
	return w, b

def cost_function(A, y):
	m = np.shape(y)[0]
	J = 0
	J = J + ((-1./m) * (np.dot(y.T, np.log(A)) + np.dot((1 - y).T, (1 - A))))
	return J

def propagate(X, y, dim, iter, l_rate):
	m = np.shape(X)[1]
	classes = np.shape(y)[1]
	costs = []
	all_W, all_b = initialize(dim, classes)

	for k in range(0, classes):
		temp_y = y[:, k].reshape(1, np.shape(y[:, k])[0])
		W = all_W[:, k].reshape(np.shape(all_W[:, k])[0], 1)
		b = all_b[0, k]

		for i in range(0, iter):

			# Forward probagation
			Z = np.dot(W.T, X) + b
			A = sigmoid(Z)
			# Save Cost
			costs.append(cost_function(A, temp_y))
			# Back probagation
			dZ = A - temp_y
			dW = (1./m) * (np.dot(X, dZ.T))
			db = (1./m) * (np.sum(dZ))
			# Update
			W = W - l_rate * dW
			b = b - l_rate * db

		W = W.reshape(np.shape(W)[0],)
		all_W[:, k] = W
		all_b[0, k] = b
	#print all_W
	#print all_b
	return all_W, all_b, costs



def predict_OnevsAll(X, W, b):
	classes = np.shape(W)[1]
	m = np.shape(X)[1]
	all_A = np.zeros((classes, m))

	for k in range(0, classes):
		Z = np.dot(W[:, k].T, X) + b[0, k]
		A = sigmoid(Z)
		all_A[k, :] = A
	print all_A
	print "----------"
	print np.amax(all_A, axis=0)
	print "----------"
	print np.argmax(all_A, axis=0)
	print "----------"


	

if __name__ == '__main__':
	X, y = transform_to_binary(train_x, train_y, 4, 10)
	W, b, costs = propagate(X, y, 4, 10000, 0.1)
	test_x = [0, 2, 3, 4, 9, 5, 6, 3, 0, 7, 1, 8, 0, 2, 3, 5, 9, 0, 1, 2, 7, 6, 6, 2, 7]
	test_y = [1, 3, 4, 5, 0, 6, 7, 4, 1, 8, 2, 9, 1, 3, 4, 6, 0, 1, 2, 3, 8, 7, 7, 3, 8]
	X, y = transform_to_binary(test_x, test_x, 4, 10)
	predict_OnevsAll(X, W, b)
	print "Real values: {}".format(test_y)  

