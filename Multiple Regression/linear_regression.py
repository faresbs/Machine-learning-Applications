"""
Linear regression 
using only numpy
"""

from math import sqrt
from random import randrange
from random import seed
import numpy as np

#calculate the mean of values
def mean(values):
    return sum(values) / float(len(values))
#calculate the variance of values
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])/len(values)

#calculate the standard deviation for the normalization
def deviation(values):
    #transform the dataset to a simple list
    values = [val for sublist in values for val in sublist]
    return sqrt(variance(values, mean(values)))


######################################################################
# 1-Extract data                                                     #  
######################################################################

#Load the csv file and extract the x and y from dataset
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


######################################################################
# 2-Split the data                                                   #  
######################################################################

#Split a dataset into a train, test and cross-validation set

    #Split a dataset into a train and test set
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


#Split data using cross validation    
#Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)  
    return dataset_split


######################################################################
# 3-Feature Normalization                                            #  
######################################################################

#Normalization by rescaling
#Rescale dataset columns to the range 0-1
def rescaling(dataset):
    #extract features from dataset
    y = list()
    for row in dataset:
        y.append(row.pop(-1))

    value_max = list()
    value_min = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min.append(min(col_values))
        value_max.append(max(col_values))
    for row in dataset:
        for i in range(len(row)):
            #if the feature is stable(max and min are equal then new feature value is set to 0.5)
            if (value_max[i] - value_min[i]) == 0:
                row[i] = 0.5
                continue
            row[i] = (row[i] - value_min[i]) / (value_max[i] - value_min[i])

    #After normalization the features we need to return the y into dataset
    i = 0
    for row in dataset:
        row.append(y[i])
        i += 1



def rescaling_model (dataset, to_predict):
        #extract features from dataset
    y = list()
    for row in dataset:
        y.append(row.pop(-1))

    value_max = list()
    value_min = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min.append(min(col_values))
        value_max.append(max(col_values))
    for row in dataset:
        for i in range(len(row)):
            #if the feature is stable(max and min are equal then new feature value is set to 0.5)
            if (value_max[i] - value_min[i]) == 0:
                row[i] = 0.5
                continue
            row[i] = (row[i] - value_min[i]) / (value_max[i] - value_min[i])

    #After normalization the features we need to return the y into dataset
    i = 0
    for row in dataset:
        row.append(y[i])
        i += 1
    
    for i in range(len(to_predict)-1):
        to_predict[i] = (to_predict[i] - value_min[i]) / (value_max[i] - value_min[i])


# FIX ME #
#Normalization by standard Deviation
def standardization(dataset):
    #extract features from dataset
#    y = list()
#    for row in dataset:
#        y.append(row.pop(-1)) 
    #calculate the mean of every feature
    feature_mean = list()
    for i in range(len(dataset[0])):
        feature = [row[i] for row in dataset]
        feature_mean.append(mean(feature))
    for row in dataset:
        for i in range(len(row)):
            #TO CHECK
            #if the feature is stable then new feature value is set to 0.5)
            if feature_mean[i] == row[i]:
                row[i] = 0.5
                continue 
            row[i] = (row[i] - feature_mean[i]) / deviation(dataset) 
    #After normalization the features we need to return the y into dataset
#    i = 0
#    for row in dataset:
#        row.append(y[i])
#        i += 1



######################################################################
# 4-Estimate Coefficients                                            #  
######################################################################

from numpy.linalg import inv
from numpy import dot, transpose

#Estimate coefficient analytically
def coefficients_analy(train):
    x = [row[:-1] for row in train] 
    y = [row[-1] for row in train]
    return dot(inv(dot(transpose(x), x)), dot(transpose(x), y))


#Estimate coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    #print coef
    return coef

######################################################################
# 5-Make a prediction with coefficients                              #  
######################################################################

#Calculate y using coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

#Calculate y using coefficients
def predict_analy(row, coefficients):
    #yhat = coefficients[0]
    yhat = 0
    print coefficients
    for i in range(len(row)-1):
        yhat += coefficients[i] * row[i]
    return yhat

def linear_regression_analy(train, test):
    predictions = list()
    coef = coefficients_numpy(train)
    for row in test:
        yhat = predict_numpy(row, coef)
        predictions.append(yhat)
    return predictions    


#Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return predictions

######################################################################
# 6-Evaluation                                                       #  
######################################################################

# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# calculate the total variation of y 
def tv(actual):
    sum_error = 0.0
    mean_y = mean(actual)
    for i in range(len(actual)):
        error = actual[i] - mean_y 
        sum_error += (error ** 2)
    mean_error = sum_error / float(len(actual))
    return mean_error

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm_cvs(dataset, algorithm, n_folds, *args):
    rescaling(dataset)
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    mean_y = list()
    test_index = list()
    i = 0
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
        mean_y.append(tv(actual))
        
        test_index = ([row[0] for row in fold])
        show(actual, predicted, test_index, mean(actual), i)
        i += 1
    return scores, mean_y

#Evaluate an algorithm using a fixed validation set
def evaluate_algorithm_fvs(dataset, algorithm, n_folds, *args):
    rescaling(dataset)




# Poucentage of y that is described by the regression line
def R2_score(rmse, mean_y):
    rmse = mean(rmse)
    mean_y = mean(mean_y)
    mse = rmse ** 2
    result = 1 - (mse / mean_y)
    if result < 0 :
        result = 0
    return result


# Plot the figure
import matplotlib.pyplot as plt

def show(actual, predictions, test_index, mean_y, j):
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title=features[j], ylabel='Y-Axis', xlabel='X-Axis')
    #ax.text(20, -2, 'Blue = actual data \nred line = predictions\ngreen line = error rate', style='italic')
    #m, b = np.polyfit(test_index, predictions, deg=12)
    #print m * x[i] + b
    #ax.plot(test_index, m * test_index + b, color='red', linewidth=1)
    
    ax.scatter(test_index, actual, color='blue',marker='o')
    ax.scatter(test_index, predictions, color='red',marker='o')
    
    #plt.axhline(y=mean_y)

    #for i in range(0, len(test_index)):
    #    ax.plot([test_index[i], test_index[i]], [actual[i], predictions[i]], color='green')
    
    plt.savefig('linear_regression%d' %j)
    #plt.show()

######################################################################
# 7-Use it on real example                                           #  
######################################################################

def model_sgd(dataset, to_predict, l_rate, n_epoch):
    rescaling_model(dataset, to_predict)
    coef = coefficients_sgd(dataset, l_rate, n_epoch)
    prediction = predict(to_predict, coef)
    return prediction   

def model_analy(dataset, to_predict):
    coef = coefficients_numpy(dataset)
    prediction = predict_numpy(to_predict, coef)
    print prediction
    return prediction   


if __name__ == '__main__':
    seed(1)
    dataset = load_csv('winequality-white.csv')

    #Evaluate model
    rmse, mean_y = evaluate_algorithm_cvs(dataset, linear_regression_sgd, 10, 0.01, 50)
    print('RMSE: %.3f' % (mean(rmse)))
    print ('R2: %.3f' % R2_score(rmse, mean_y))

    #test it on a real example
    to_predict = [6.4,0.31,0.38,2.9,0.038,19,102,0.9912,3.17,0.35,11,7]
    #rescaling_model(dataset, to_predict)
    #print model_sgd(dataset, to_predict, 0.001, 50)
    #model_analy(dataset, to_predict)


