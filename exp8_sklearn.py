import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from sklearn import linear_model
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

    train_features = shape_csv('train_features.csv')
    train_labels = shape_csv('train_labels.csv')
    valid_features = shape_csv('valid_features.csv')
    valid_labels = shape_csv('valid_labels.csv')

    model = linear_model.SGDRegressor(loss='squared_loss', alpha=0.02, learning_rate='constant')
    model.fit(train_features, train_labels)

    valid_pred = model.predict(valid_features)

    valid_labels, valid_pred = zip(*sorted(zip(valid_labels, valid_pred)))

    pred.plot(range(0, len(valid_pred)),valid_pred, 'b.', label="Predicted")
    pred.plot(range(0, len(valid_labels)), valid_labels, 'r.', label="Target")
    pred.legend()
    fig_pred.show()
    fig_pred.savefig('sklearn.png')


if __name__ == "__main__":
    main()




