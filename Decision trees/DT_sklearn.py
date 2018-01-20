import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

import graphviz

np.random.seed(1)

# Load dataset
iris = load_iris()

# Split data to train and test sets
train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)

# Remove 3 examples for testing
#test_idx = [0, 50, 100]

#Training data
#train_y = np.delete(iris.target, test_idx)
#train_X = np.delete(iris.data, test_idx, axis=0)

#Testing data
#test_X = iris.data[test_idx]
#test_y = iris.target[test_idx]

# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train it with the training data
clf = clf.fit(train_X, train_y)

# Test the model with our testing data
y_pred = clf.predict(test_X)

# The perfect performance is 1.0 and the worst is 0.0
print "Model accuracy : %f" %accuracy_score(test_y, y_pred)

# Visualize decision tree graph using graphviz
dot_data = tree.export_graphviz(clf, out_file=None, class_names=iris.target_names) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

# Test out the graph
print test_y[1]
print test_X[1]
print test_X[1][2]
print iris.target_names	