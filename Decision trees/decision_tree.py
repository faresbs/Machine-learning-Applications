"""
Decision Tree using Python
Training data
"""
import math

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


def split(data, question):

    true_rows, false_rows = [], []
    for row in data:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# Count number of different elements in examples
def counting(data):
	counts = {}
	for row in data:
		label = row[-1]
		if(label not in counts):
			counts[label] = 0
		counts[label] += 1
	
	return counts


def Gini_impurity(data):
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
def info_gain(child_left, child_right, parent_uncertainty):

    n = len(child_left) + len(child_right)
    average_children_uncertainty = (float(len(child_left)) / n) * Gini_impurity(child_left) + (float(len(child_right)) / n) * Gini_impurity(child_right)   
    return parent_uncertainty - average_children_uncertainty



if __name__ == '__main__':
	#print Question(0, 'Red')
	q = Question(0, 'Red')
	#print training_data[3]
	#print q.match(training_data[3])

	trues, falses = split(training_data, q)
	parent = Gini_impurity(training_data)
	print trues
	print falses
	print parent
	print info_gain(trues, falses, parent)