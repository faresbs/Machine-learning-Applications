# Deep Neural networks 
# CIFAR-10 dataset

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import cPickle


np.random.seed(1)


# Read Python "pickled" image object in batches 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# Tranform to proper format for visualization
def transform_data(data):
	
	X = data["data"]
	y = data["labels"]
	# Reshape y to (m, 1) from (m,)
	y = np.reshape(y, (np.shape(y)[0], 1))
	X = np.split(X, 3, axis=1)
	X = np.reshape(X, (3, 10000, 32, 32))
	# Rearranged in the order 1, 2, 0 so that we could display it with Image.fromarray
	X = np.transpose(X, (1, 2, 3, 0))

	return X, y


# Transform labels y to binary 0s and 1s -> 0100000000
def transform_labels(y, classes):
	m = np.shape(y)[0]
	bin_y = np.zeros((m, classes))
	temp = np.zeros((classes, ))
	for i in range(0, m):
		temp[y[i]] = 1
		bin_y[i, :] = temp
		temp[y[i]] = 0
	return bin_y


# Activation functions
def relu(Z):
	A = np.maximum(0, Z)
	return A

def softmax(Z):
	t = np.exp(Z)
	A = t / np.sum(t, axis=0)
	return A


def initialize_parameters(layer_dims):
    
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # Check if dimensions are correct
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters



def forward_propagation(A_prev, W, b, activation):

	if(activation == "relu"):
		# Forward pass
		Z = np.dot(W, A_prev) + b
		A = relu(Z)
		# Check Z dim
		assert(Z.shape == (W.shape[0], A_prev.shape[1]))

	if(activation == "softmax"):
		# Forward pass
		Z = np.dot(W, A_prev) + b
		A = softmax(Z)
		# Check Z dim
		assert(Z.shape == (W.shape[0], A_prev.shape[1]))

	return A, Z


# FIX: WHAT YOU DO WITH DIFFERENT SIZED LAYERS

def forward_model(X, parameters):

	caches = []
	L = len(parameters) / 2
	A = X
	for l in range(1, L):
		A_prev = A

		A, Z = forward_propagation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
		
		cache = (A, Z)
		print np.shape(cache)
		caches.append(cache)

	# Apply Softmax on the last layer
	AL, Z = forward_propagation(A_prev, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")

	cache = (AL, Z)
	caches.append(cache)

	# AL needs to be the same shape as y (10, m)
	assert(AL.shape[1] == X.shape[1])
	return AL, caches



def cost_function(AL, y):
	m = np.shape(y)[0]
	# Reshape y to be size of (classes, m) like AL
	y = y.T
	# Categorical Cross entropy for softmax
	J = 0
	print (-1./m) * (np.sum(np.sum(np.multiply(y, np.log(AL)), axis=0)))
	return J



def backward_prop(dA, cache, cached_parameters, activation):
	W, b = cached_parameters
	A, Z = cache
	
	if(activation = "relu"):
		if(Z > 0):
			dZ = 1
		elif
			dZ = 0
	# Fix Softmax	
	elif(activation = "softmax"):
		dZ = 


	dW = (1./m) * np.dot(dZ, A.T)
    db = (1./m) * np.squeeze(np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(parameters[0].T, dZ)


	return	dA_prev, dW, db



batch_1 = unpickle("cifar-10-batches-py/data_batch_1")

train_X_batch_1 = batch_1["data"]
train_y_batch_1 = batch_1["labels"]

n = np.shape(train_X_batch_1)[1]

y = transform_labels(train_y_batch_1, 10)

# Reshape X from (m, n) to (n, m)
train_X_batch_1 = train_X_batch_1.T

# 3 hidden layers
# First layer is the input layer, the last layer is the output layer with 10 classes
layer_dims = [n, 4, 4, 4, 10]
parameters = initialize_parameters(layer_dims)
AL, cache = forward_model(train_X_batch_1, parameters)

cost_function(AL, y)




