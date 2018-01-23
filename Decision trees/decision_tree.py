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
]

# Column labels.
features_names = ["color", "diameter", "label"]


def is_numeric(value):
    #Test if a value is numeric.
    #If it's an int or float
    return isinstance(value, int) or isinstance(value, float)


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

"""
# Split to trues and falses
def binary_split(data, question):

    true_rows, false_rows = [], []
    for row in data:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
"""

# Split data according to feature
def split(data, feature):
	splits = []
	feature_values = data[:, feature]
	# Eliminate Repetitive values
	feature_values = list(set(feature_values))
	for value in feature_values:
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


# Calculate the information gain
# CHECK / 3 options for color(green, red, yellow) not just 2 (true / false)
"""
def info_gain(child_left, child_right, parent_uncertainty):

    n = len(child_left) + len(child_right)
    average_children_uncertainty = (float(len(child_left)) / n) * Gini_impurity(child_left) + (float(len(child_right)) / n) * Gini_impurity(child_right)   
    return parent_uncertainty - average_children_uncertainty
"""

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



# Return the feature with lowest gini index
def gini_index(m, *args):
	minimum = 10

	for arg in args:
		split = arg[0]
		gini = average_gini(split, m)
		if(gini < minimum):
			minimum = gini
			feature_to_choose = arg[1]

	return feature_to_choose



if __name__ == '__main__':
	#print Question(0, 'Red')
	q = Question(0, 'Red')
	#print training_data[3]
	#print q.match(training_data[3])

	#trues, falses = split(training_data, q)
	#parent = Gini_impurity(training_data)
	#print parent

	data = np.asarray(training_data)
	
	split1 = split(data, 0)
	split2 = split(data, 1)

	print split2[0]

	parent = gini_impurity(data)
	m = len(data)

	print average_gini(split1[0], m)
	print average_gini(split2[0], m)

	print features_names[gini_index(m, split1, split2)]


