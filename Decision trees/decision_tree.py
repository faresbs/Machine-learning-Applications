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
    ['Red', 0.5, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    ['Green', 1, 'Grape']
]

# Column labels.
features_names = ["color", "diameter", "label"]


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


# Split data according to feature
# Every split = (data, feature)
def split(data, feature):
	splits = []
	feature_values = data[:, feature]
	# Eliminate Repetitive values
	feature_values = list(set(feature_values))
	for value in feature_values:
		if(is_numeric(value)):
			split = data[data[:, feature] >= value]
			splits.append(split)

		else:
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
	for count in counts:
		if(count >= maximum):
			maximum = count

	return maximum


class Leaf:
	def __init__(self, data):
		self.predictions = counting(data)
	
	def __repr__(self):
		return "Leaf: %s" % (str(self.predictions))	



def build_tree(data):
	nodes = []
	m = len(data)
	parent_uncertainty = gini_impurity(data)
	feature, children = find_best_split(data)

	gain = info_gain(children, parent_uncertainty, m)
	
	# Test if gain = 0 then it's a leaf
	if(gain == 0):
		"""
		result = most_frequent(data)
		current_node = feature, result, parent_uncertainty
		nodes.append(current_node)
		"""
		return Leaf(data)

	# Call build tree function recursively for every child node
	for child_node in children:
		sub_nodes = build_tree(child_node)
	

	return feature, sub_nodes


def classify(example, node):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(example):
        return classify(example, node.true_branch)
    else:
        return classify(exaple, node.false_branch)




if __name__ == '__main__':
	#print Question(0, 'Red')
	#q = Question(0, 'Red')
	#print training_data[3]
	#print q.match(training_data[3])

	#trues, falses = split(training_data, q)
	#parent = Gini_impurity(training_data)
	#print parent

	data = np.asarray(training_data)
	
	"""
	m = len(data)
	parent = gini_impurity(data)
	feature, children = find_best_split(data)
	data1 = children[2] 
	info_gain(children, parent, m)

	m = len(data1)
	parent1 = gini_impurity(data1)
	feature, children1 = find_best_split(data1)
	print parent1
	print info_gain(children1, parent1, m)

	#find_best_split(data)
	"""

	my_tree = build_tree(data)
	print my_tree
