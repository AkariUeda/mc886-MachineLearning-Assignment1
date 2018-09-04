import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from subprocess import call

warnings.filterwarnings('ignore')

def normal_equation(X, train_labels, theta_size):
    Xt = np.transpose(X)
    Xt.shape, X.shape
    inv =np.dot(Xt, X)
    inv = inv.astype(np.float32)
    inv = linalg.inv(inv)
    theta_ne = np.dot(inv, Xt)
    theta_ne = np.dot(theta_ne, np.array(train_labels))
    for i in range(0,theta_size):
        print('Theta {:<1d} = {:<10f}'.format(i, theta_ne[i][0]))
    return theta_ne

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
    train_features = shape_csv('train_features.csv')
    train_labels = shape_csv('train_labels.csv')
    valid_features = shape_csv('valid_features.csv')
    valid_labels = shape_csv('valid_labels.csv')

    X_features = np.concatenate((train_features, valid_features), axis=0)
    X_labels = np.concatenate((train_labels, valid_labels), axis=0)
    print(X_features.shape)

    test_features = shape_csv('test_features.csv')
    test_labels = shape_csv('test_labels.csv')

    theta = normal_equation(X_features, X_labels, 10)


    h = np.dot(test_features, theta)

    J = np.dot(test_features, theta)
    J = np.subtract(J,test_labels)
    cost = np.sum(np.square(J))/(2*len(test_features))
    print("Prediction error: "+str(cost))
    test_labels, H = zip(*sorted(zip(test_labels, h)))

    pred.plot(range(0, len(H)),H, 'b.', label="Predicted")
    pred.plot(range(0, len(test_labels)), test_labels, 'r.', label="Target")

    pred.legend()

    fig_pred.show()
    fig_pred.savefig('normal_equation.png')

if __name__ == "__main__":
    main()




