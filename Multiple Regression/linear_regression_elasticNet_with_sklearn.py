from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from matplotlib import pyplot as plt 

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
    
    e = ElasticNet(alpha=0.1)
    e.fit(x, y)
    ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

    y_predicted = e.predict(x)
    mse = mean_squared_error(y, y_predicted)
    print('RMSE: %.3f' % (sqrt(mse)))
    r2 = r2_score(y, y_predicted)
    print ('R2: %.3f' % r2)


       #create multiple variables for features
    x_plot = [None] * len(x[0])
    for i in range(len(x[0])):
        x_plot[i] = [row[i] for row in x]


    #Plot all the features
    colors = ['teal', 'yellowgreen', 'gold', 'hotpink', 'lightskyblue',  'lime', 'magenta', 'cyan', 'firebrick', 'royalblue', 'orange']

    for i in range(len(x[0])):
        plt.scatter(x_plot[i], y_predicted, color=colors[i], s=30, marker='o', label="Predicted point")
        plt.scatter(x_plot[i], y, color='navy', s=30, marker='o', label="Actual point")
        
        #m, b = np.polyfit(x_0, y_predicted, deg=1)
        #print m * x[i] + b
        #plt.plot(x[i], int(m) * x_0 + b, color="gold", linewidth=2,)
        
        plt.legend(loc='lower left')
        plt.title("Feature %s" %i)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.show()

    #Test it on a real example
    to_predict = [6.4,0.31,0.38,2.9,0.038,19,102,0.9912,3.17,0.35,11]
    print "Your prediction : %d" %e.predict(to_predict)
