"""
Polynomial regression using sklearn
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

#supress all matplotlib warnings
import warnings
warnings.filterwarnings("ignore")


def load_csv(filename):
    #dataset is a list of lists
    dataset = list()
    file = open(filename, 'rb')
    Lines = file.readlines()
    for line in Lines:
        #strip characters from beginning and the end of the string
        line = line.strip()
        listFromLine = line.split(';')
        for i in range(len(listFromLine)):
            listFromLine[i] = float(listFromLine[i])
        dataset.append(listFromLine)
    return dataset


if __name__ == '__main__':

    filename = 'winequality-white.csv'
    dataset = load_csv(filename)

    x = list()
    y = list()

    for row in dataset:
        x.append(row[:-1])
        y.append(row[-1])
    
    
    #create multiple variables for features
    x_plot = [None] * len(x[0])
    for i in range(len(x[0])):
        x_plot[i] = [row[i] for row in x]


    colors = ['teal', 'yellowgreen', 'gold', 'hotpink', 'lightskyblue',  'lime']  

    for i in range(len(x[0])):
    	for count, degree in enumerate([2, 3, 4]):
    		model = make_pipeline(PolynomialFeatures(degree), Ridge())
    		model.fit(x, y)
    		y_pred = model.predict(x)

    		plt.scatter(x_plot[i], y_pred, color=colors[count], s=30, marker='o', label="Predicted point With degree %d" %degree)
        
    	plt.scatter(x_plot[i], y, color='navy', s=30, marker='o', label="Actual point")	
        plt.legend(loc='lower left')
    	plt.title("Feature %s" %i)
    	plt.ylabel('Y')
    	plt.xlabel('X')
    	plt.show()	



    for count, degree in enumerate([2, 3, 4, 5]):
    		model = make_pipeline(PolynomialFeatures(degree), Ridge())
    		model.fit(x, y)
    		y_pred = model.predict(x)

    		#Evaluate the regression line
    		mse = mean_squared_error(y, y_pred)
    		print('Polynomial Degree: %d' %degree) 
    		print('RMSE: %.3f' % (sqrt(mse)))
    		r2 = r2_score(y, y_pred)
    		print ('R2: %.3f' % r2)

    		to_predict = [6.4,0.31,0.38,2.9,0.038,19,102,0.9912,3.17,0.35,11]
    		print "Your prediction : %d" %model.predict(to_predict)
    		print "-----------------------------------------------------"
    


    


