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
	J = J + (-1./m) * (np.sum(np.sum(np.multiply(y, np.log(AL)), axis=0)))
	return J



def backward_prop(y, cache, cached_parameters, activation):
	
	# Retrive cached values from forward prop
	W, b = cached_parameters
	A, Z, A_prev = cache

	# Reshape y to be size of (classes, m) like dAL
	y = y.T

	m = np.shape(y)[1]

	# dZ (same shape as Z) to use for relu
	dZ = Z
	
	if(activation == "relu"):
		dZ[dZ > 0] = 1
		dZ[dZ <= 0] = 0

	elif(activation == "softmax"):
		dZ = A - y

	dW = (1./m) * np.dot(dZ, A_prev.T)
	db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
	
	dA_prev = np.dot(W.T, dZ)

	# gradients same dim as the cached values
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	assert (dA_prev.shape == A_prev.shape)

	return	dA_prev, dW, db



def backward_model(y, X, caches, parameters):
	grads = {}
	L = len(caches) 
	m = AL.shape[1]

	# caches is with the shape (4, 2) : for 4 layers | caches[0] for first layer : 2 (A and Z), 4 nodes and 10000 examples

	# Start with last layer    
	caches_parameters = (parameters["W" + str(L)], parameters["b" + str(L)])
	# cache = (AL, ZL, A2)
	cache = (caches[L - 1][0], caches[L - 1][1], caches[L - 2][0])
	# Gradients for the last layer
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = backward_prop(y, cache, caches_parameters, "softmax")

	for l in reversed(range(1, L)):
		print l
		caches_parameters = (parameters["W" + str(l)], parameters["b" + str(l)])

		# last layer A_prev takes the input layer X
		if (l == 1):
			A_prev = X
		else:
			A_prev = caches[l - 2][0]
		# cache = (AL, ZL, A_prev)
		cache = (caches[l - 1][0], caches[l - 1][1], A_prev)

		dA_prev_temp, dW_temp, db_temp = backward_prop(y, cache, caches_parameters, "relu")
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l)] = dW_temp
		grads["db" + str(l)] = db_temp

	for key, value in grads.items():
		print key, value.shape

	return grads




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
AL, caches = forward_model(train_X_batch_1, parameters)

cost_function(AL, y)

backward_model(y, train_X_batch_1, caches, parameters)

#caches_parameters = (parameters["W4"], parameters["b4"])
# cache = (AL, ZL, A3)
#cache = (caches[3][0], caches[1], caches[2][0])
#backward_prop(y, cache, caches_parameters, "softmax")


