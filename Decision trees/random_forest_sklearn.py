"""
Random forest using sklearn
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

import graphviz

import time
start_time = time.time()

np.random.seed(1)

# Load dataset
iris = load_iris()

# Split data to train and test sets
train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)


rf = RandomForestClassifier(n_estimators=10)
rf.fit(train_X, train_y)

print(test_y)
print(rf.predict(test_X))
y_pred = rf.predict(test_X)
print "Model accuracy : %f" %accuracy_score(test_y, y_pred)

print("--- %s seconds ---" % (time.time() - start_time))