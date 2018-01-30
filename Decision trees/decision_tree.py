"""
Decision Tree using Python
Training data
"""
import math
import numpy as np


training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    ['Green', 1, 'Grape'],
]

features_names = ["color", "diameter", "label"]
features_index = {"color":0, "diameter":1, "label":2}

def is_numeric(value):
    #Test if a value is numeric.
    #If it's a float else throw an exception
    try:
        float(value)
        return True
    except ValueError:
        return False


class Question:

    def __init__(self, feature, feature_value):
        self.feature = feature
        self.feature_value = feature_value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.feature]
        if is_numeric(val):
            return val >= self.feature_value
        else:
            return val == self.feature_value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.feature_value):
            condition = ">="
        return "Is %s %s %s?" % (
            features_names[self.feature], condition, str(self.feature_value))


# Split to trues and falses
def binary_split(data, question):

    true_rows, false_rows = [], []
    for row in data:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

# TO FIX 
# ONly works with discrete values

# Split data according to feature
# Every split = (data, feature)
def split(data, feature):
	splits = []
	feature_values = data[:, feature]
	# Eliminate Repetitive values
	feature_values = list(set(feature_values))
	for value in feature_values:
		"""
		if(is_numeric(value)):
			split = data[data[:, feature] >= value]
			splits.append(split)

		else:
		"""
		split = data[data[:, feature] == value]
		splits.append(split)


	return splits, feature


# Count number of different elements in examples
def counting(data):
	counts = {}
	for row in data:
		label = row[-1]
		if(label not in counts):
			counts[label] = 0
		counts[label] += 1
	
	return counts


def gini_impurity(data):
	counts = counting(data)
	# Number of elements
	# Must be converted to float
	n = float(len(data))
	
	if(n == 1):
		return 0
	
	somme = 0
	for elem in counts:
		somme += (counts[elem] / n)**2
	impurity = 1 - somme
	return impurity


# FIX this / CHECK this / Gives result of > 1
def entropy(data):
	counts = counting(data)
	logs = []
	# Number of elements
	# Must be converted to float
	n = float(len(data))

	if(n == 1):
		return 0
	
	entropy = 0
	for elem in counts:
		pi = counts[elem] / n
		log = math.log(pi, 2)
		entropy -= pi * log
	
	return entropy


 
def info_gain(child_nodes, parent_uncertainty, m):

    average_children_uncertainty = 0
    for node in child_nodes:
    	average_children_uncertainty = (float(len(node)) / m) * gini_impurity(node)
    
    return parent_uncertainty - average_children_uncertainty



# Calculate the average of the child nodes   	
def average_gini(split, m):
	
	# number of childs
	s = len(split)
	gini = 0.0
	for i in range(s):
		# Number of examples for a given node
		n = float(len(split[i]))
		gini = gini + (gini_impurity(split[i]) * (n / m))

	return gini



# Return the feature with lowest gini 
def gini_index(m, args):
	minimum = 10
	for arg in args:
		split = arg[0]
		gini = average_gini(split, m)
		if(gini < minimum):
			minimum = gini
			feature_to_choose = arg[1]

	return feature_to_choose


def find_best_split(data):

	m = len(data)
	nb_features = np.shape(data)[1] - 1
	splits = []
	averages = []
	for i in range(nb_features):
		splits.append(split(data, i))

	feature = features_names[gini_index(m, splits)]
	child_nodes = splits[gini_index(m, splits)][0]

	return feature, child_nodes


# return the most recurent element
def most_frequent(data):
	# count elements with diferent labels
	counts = counting(data)
	maximum = 0
	for label in counts:
		if(counts[label] >= maximum):
			maximum = counts[label]
			most_frequent = label

	return most_frequent


class Leaf:
	def __init__(self, data):
		self.predictions = counting(data)
	
	def __repr__(self):
		return "%s" % (str(self.predictions))	



class Decision_Node:
    def __init__(self, feature, sub_nodes, data):
        self.feature = feature
        self.feature_value = data[0, features_index[str(feature)]]
        self.sub_nodes = sub_nodes
        self.data = data



def build_tree(data):

	nodes = []

	m = len(data)
	parent_uncertainty = gini_impurity(data)
	feature, children = find_best_split(data)

	gain = info_gain(children, parent_uncertainty, m)
	
	# Test if gain = 0 then it's a leaf
	if(gain == 0):
		nodes.append(Leaf(data))
		return nodes

	# Call build tree function recursively for every child node
	for child_node in children:
		sub_nodes = build_tree(child_node)

		nodes.append(Decision_Node(feature, sub_nodes, child_node))

	return nodes


# TO FIX
def print_tree(nodes):
	for node in nodes:
		if isinstance(node, Leaf):
			print "Leaf"
			print (node.predictions)
			return

		print (node.data)
		print ('--> Feature:')
    	print (node.feature)
    	print (node.feature_value)
    	print ('--> Children:')
    	print_tree(node.sub_nodes)





def question(feature, feature_value, example):
	return


def predict(example, nodes):
	for node in nodes:
		question = Question(features_index[str(node.feature)], node.feature_value)

		if isinstance(node, Leaf):
			return node.predictions

        if question.match(example):
        	return predict(example, node.sub_nodes)




if __name__ == '__main__':

	data = np.asarray(training_data)
	my_tree = build_tree(data)
	print predict(training_data[3], my_tree)
	
	"""
	print data
	print my_tree
	print my_tree[0].data
	print my_tree[0].feature
	print my_tree[0].feature_value
	print my_tree[0].sub_nodes
	
	print my_tree[1].data
	print my_tree[1].feature
	print my_tree[1].feature_value
	print my_tree[1].sub_nodes

	print my_tree[1].sub_nodes[0].feature
	print my_tree[1].sub_nodes[0].feature_value
	print my_tree[1].sub_nodes[0].sub_nodes
	print my_tree[1].sub_nodes[1].feature
	print my_tree[1].sub_nodes[1].feature_value
	print my_tree[1].sub_nodes[1].sub_nodes
"""
	

	#if isinstance(my_tree[1].sub_nodes[0].sub_nodes[0], Leaf):
	#	print "true"

	#print_tree(my_tree)

