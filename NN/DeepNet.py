# Deep Neural networks 
# Using the CIFAR-10 dataset

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import cPickle


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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
	A = t / np.sum(t)
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
		Z = np.dot(W, A_prev.T) + b
		A = relu(Z)
		# Check Z dim
		assert(Z.shape == (W.shape[0], A_prev.shape[0]))

	if(activation == "softmax"):
		# Forward pass
		Z = np.dot(W, A_prev.T) + b
		A = softmax(Z)
		# Check Z dim
		assert(Z.shape == (W.shape[0], A_prev.shape[0]))		

	return A, Z


# MAYBE WE NEED JUST ONE LOOP OVER THE FORWARD BACKWARD PASS
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

	# AL needs to be the same shape as y (m, 10)
	assert(AL.shape[0] == X.shape[1])

	return AL, caches



def cost_function(AL, y):
	m = np.shape(y)[0]
	# Cross entropy
	J = 0
	J = J + ((-1./m) * (np.dot(y.T, np.log(AL)) + np.dot((1 - y).T, np.log(1 - AL))))
	J = np.squeeze(J)
	return J



def backward_prop():
	return



batch_1 = unpickle("cifar-10-batches-py/data_batch_1")

train_X_batch_1 = batch_1["data"]
train_y_batch_1 = batch_1["labels"]

n = np.shape(train_X_batch_1)[1]

transform_labels(train_y_batch_1, 10)

# Create DeepNet with 2 hidden layers
# First layer is the input layer
layer_dims = [n, 4, 4, 4]
parameters = initialize_parameters(layer_dims)
forward_model(train_X_batch_1, parameters)




