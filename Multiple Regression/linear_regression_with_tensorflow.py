"""
Linear regreression 
with Tensorflow
"""

import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk 
import matplotlib.pyplot as plt


LOGDIR = 'tmp/reg'

#Load the csv file and extract the x and y from dataset
def load_csv(filename):
    x = list()
    y = list()
    file = open(filename, 'rb')
    Lines = file.readlines()
    for line in Lines:
        #Strip characters from beginning and the end of the string
        line = line.strip()
        listFromLine = line.split(';')
        for i in range(len(listFromLine)):
        	listFromLine[i] = float(listFromLine[i])
        y.append(listFromLine.pop())
       	x.append(listFromLine)
    x = np.array(x)
    y = np.array(y)
    return x, y

#Normalize features by standardization using standard derivation
def feature_normalize(features):
    mean = np.mean(features,axis=0)
    derivation = np.std(features,axis=0)
    return (features - mean)/derivation


def train_test_split(features, labels, split):	
    x_train, x_test, y_train, y_test = sk.train_test_split(features, labels,test_size=split, random_state = 42)
    return x_train, x_test, y_train, y_test


def predict(W, b, to_predict):
	to_predict = feature_normalize(to_predict)
	coef = np.array(W)
	coef = coef.flatten('F')

	result = np.zeros(shape=(coef.shape[0]))
	for i in range(len(to_predict)):
		result[i] = (coef[i] * to_predict[i])

	prediction = np.sum(result) + b
	#print prediction
	return prediction


def linear_regression(dataset, to_predict, n_epoch, l_rate):
	features, labels = load_csv(dataset)
	normalized_features = feature_normalize(features)

	labels = np.reshape(labels, [labels.shape[0], 1])

	n = features.shape[0]
	r = features.shape[1]

	X = tf.placeholder("float", [None, r], name="feature")
	Y = tf.placeholder("float", [None, 1], name="label")
	
	W = tf.Variable(tf.ones([r, 1]), name="weight")
	tf.summary.histogram("weights", W)
	
	b = tf.Variable(tf.ones(1), name="bias")
	tf.summary.histogram("baises", b)

	init = tf.global_variables_initializer()

	y = tf.matmul(X, W) + b

	with tf.name_scope("loss"):
		loss = tf.reduce_mean(tf.square(y - Y))
		tf.summary.scalar('Loss', loss)

	optimizer = tf.train.AdamOptimizer(l_rate)
	train = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	sess = tf.Session()


	#Visualize on Tensorboard
	#Store in the LOGDIR directory
	#tensorboard --logdir=path/to/log-directory
	
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(LOGDIR)
	writer.add_graph(sess.graph)

	sess.run(init)

	for epoch in range(n_epoch):
		sess.run(train, feed_dict={X:normalized_features, Y:labels})
		s = sess.run(merged, feed_dict={X: normalized_features, Y:labels})
		writer.add_summary(s, epoch)
		print "Iteration: {0} ---- Prediction: {1:0.1f}".format(epoch, predict(sess.run(W), sess.run(b), to_predict)[0])
	return predict(sess.run(W), sess.run(b), to_predict)
	


if __name__ == '__main__':
	#Test our model on a real example
	to_predict = np.array([6.4,0.31,0.38,2.9,0.038,19,102,0.9912,3.17,0.35,11])
	linear_regression("winequality-white.csv", to_predict, 500, 0.01)




