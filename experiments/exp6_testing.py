import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from sklearn import metrics
from subprocess import call

warnings.filterwarnings('ignore')

def shape_csv(name):
    file = pd.read_csv(name,header=None)    
    file = np.array(file)
    file = file.astype(np.float)
    return file    

def main():

    #Plot settings
    matplotlib.style.use('seaborn')

    fig_pred = plt.figure()
    pred = fig_pred.add_subplot(1,1,1)
    pred.set_ylabel('Price')
    pred.set_xlabel('Examples')

    cores = ['tab:blue', 'tab:orange']

    test_features = shape_csv('test_features.csv')
    test_labels = shape_csv('test_labels.csv')

    theta = shape_csv('csv/final_theta.csv')
    h = np.dot(test_features, theta[0])
    cost = metrics.mean_squared_error(test_labels, h)
    print("Prediction error: "+str(cost))
    test_labels, H = zip(*sorted(zip(test_labels, h)))

    pred.plot(range(0, len(H)),H, 'b.', label="Predicted")
    pred.plot(range(0, len(test_labels)), test_labels, 'r.', label="Target")

    pred.legend()

    fig_pred.show()
    fig_pred.savefig('testing.png')

if __name__ == "__main__":
    main()




