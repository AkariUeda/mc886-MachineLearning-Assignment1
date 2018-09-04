import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
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

    #train_lr(theta, train_features, train_labels, iterations, alpha)
    theta = shape_csv('final_theta.csv')

    h = np.dot(test_features, theta[0])

    J = np.dot(test_features, theta[0])
    J = np.subtract(J,test_labels)
    cost = np.sum(np.square(J))/(2*len(test_features))
    print("Prediction error: "+str(cost))
    test_labels, H = zip(*sorted(zip(test_labels, h)))

    pred.plot(range(0, len(H)),H, 'b.', label="Predicted")
    pred.plot(range(0, len(test_labels)), test_labels, 'r.', label="Target")

    pred.legend()

    fig_pred.show()
    fig_pred.savefig('testing.png')

if __name__ == "__main__":
    main()




