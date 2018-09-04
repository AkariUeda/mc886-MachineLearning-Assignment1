import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from sklearn import linear_model
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

    train_features = shape_csv('csv/train_features.csv')
    train_labels = shape_csv('csv/train_labels.csv')
    valid_features = shape_csv('csv/valid_features.csv')
    valid_labels = shape_csv('csv/valid_labels.csv')
    test_features = shape_csv('csv/test_features.csv')
    test_labels = shape_csv('csv/test_labels.csv') 

    train_features = np.concatenate((train_features, valid_features), axis=0)
    train_labels = np.concatenate((train_labels, valid_labels), axis=0)
 
    model = linear_model.SGDRegressor(loss='squared_loss', alpha=0.0001, learning_rate='constant')
    model.fit(train_features, train_labels)

    test_pred = model.predict(test_features)
    cost = metrics.mean_squared_error(test_labels, test_pred)
    print("Custo predito: "+str(cost))
    test_labels, test_pred = zip(*sorted(zip(test_labels, test_pred)))

    pred.plot(range(0, len(test_pred)),test_pred, 'b.', label="Predicted")
    pred.plot(range(0, len(test_labels)), test_labels, 'r.', label="Target")
    pred.legend()
    fig_pred.show()
    fig_pred.savefig('sklearn.png')

if __name__ == "__main__":
    main()




