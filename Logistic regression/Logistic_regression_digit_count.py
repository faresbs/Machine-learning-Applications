"""
Logistic regression to teach the machine how to count 
X = 1 -> y = 2
X = 5 -> y = 6
X = 9 -> y = 0
using python 2.7 / Numpy and matplotlib for visualization 
"""

import numpy as np
import matplotlib as plt

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

# Activation function
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
	m = np.shape(y)[1]
	J = 0
	# Reshape A and y from (1, 12) to (12, 1) : y and A must have the same shape
	y = y.T
	A = A.T
	J = J + ((-1./m) * (np.dot(y.T, np.log(A)) + np.dot((1 - y).T, np.log(1 - A))))
	
	J = float(J)
	J = float("{0:.3f}".format(J))

	return J

def propagate(X, y, dim, iter, l_rate):
	m = np.shape(X)[1]
	classes = np.shape(y)[1]
	costs = np.zeros((10, classes))
	cost = []
	all_W, all_b = initialize(dim, classes)

	for k in range(0, classes):
		temp_y = y[:, k].reshape(1, np.shape(y[:, k])[0])
		W = all_W[:, k].reshape(np.shape(all_W[:, k])[0], 1)
		b = all_b[0, k]

		for i in range(iter):

			# Forward probagation
			Z = np.dot(W.T, X) + b
			A = sigmoid(Z)

			# Save Cost
			# VIEW: COST OF EVERY CLASS OR AVERAGE COST OF THE CLASSES
			if i % 100 == 0:
				cost.append(cost_function(A, temp_y))

			# Back probagation
			dZ = A - temp_y
			dW = (1./m) * (np.dot(X, dZ.T))
			db = (1./m) * (np.sum(dZ))

			# Update
			W = W - l_rate * dW
			b = b - l_rate * db

		costs[:, k] = cost
		cost = []

		W = W.reshape(np.shape(W)[0],)
		all_W[:, k] = W
		all_b[0, k] = b

	#print all_W
	#print all_b

	# Average the cost on all the classes for every iteration
	costs = np.mean(costs, axis=1)

	return all_W, all_b, costs



def predict(a, W, b, dim):
	all_A =[]
	classes = np.shape(W)[1]
	bin_a = np.zeros((dim, 1))
	# Convert a to binary
	string = np.binary_repr(a, width=dim)

	for j in range (0, dim):
		bin_a[j] = string[j]

	for k in range(0, classes):
		Z = np.dot(W[:, k].T, bin_a) + b[0, k]
		A = sigmoid(Z)
		A = float(A)
		# Limiting to 3 decimal points
		A = float("{0:.3f}".format(A))
		all_A.append(A)
	# print all_A

	# Choose the index of the one with the highest probability
	# print np.argmax(all_A)

	return np.argmax(all_A)



def predict_testSet(X, W, b):
	classes = np.shape(W)[1]
	m = np.shape(X)[1]
	all_A = np.zeros((classes, m))

	for k in range(0, classes):
		Z = np.dot(W[:, k].T, X) + b[0, k]
		A = sigmoid(Z)

		# Limiting to 3 decimal points
		A = [float("{0:.3f}".format(i)) for i in A]
		all_A[k, :] = A

	#print all_A
	#print "----------"
	#print np.amax(all_A, axis=0)
	#print "----------"
	#print np.argmax(all_A, axis=0)
	return np.argmax(all_A, axis=0)



# Count from a to c (you can't exceed 9 because the machine has learned how to to count until 9)
# a is the starting point and c is the end point

# FIX ME 

def count(a, c, W, b, dim):
	
	numbers = []
	i = predict(a, W, b, dim)

	if(a == c):
		return a

	if(a < c):
		numbers.append(a)
		numbers.append(i)
		while (i != c):
			i = predict(i, W, b, dim)
			numbers.append(i)
	return numbers


if __name__ == '__main__':

	train_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	train_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

	test_x = [0, 2, 3, 4, 9, 5, 9, 1]
	test_y = [1, 3, 4, 5, 0, 6, 0, 2]

	X, y = transform_to_binary(train_x, train_y, 4, 10)
	W, b, costs = propagate(X, y, 4, 1000, 0.1)

	X, y = transform_to_binary(test_x, test_y, 4, 10)
	print "Predicted Values: {}".format(predict_testSet(X, W, b))
	print "Real values: {}".format(test_y)  
	print "Cost: {}".format(costs)
	#print count(0, 9, W, b, 4)

	

