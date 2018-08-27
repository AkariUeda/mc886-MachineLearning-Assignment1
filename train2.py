import numpy as np
from numpy import linalg
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from subprocess import call

warnings.filterwarnings('ignore')

def normalize_features(features):
  for f in range(1, features.shape[1]):
    mean = features[:,f].mean()
    features[:,f] = np.subtract(features[:,f],mean)
    max_value = features[:,f].max()
    features[:,f] = np.divide(features[:,f],max_value)
  return features

def train_lr(theta, X, Y, iteracoes, alpha):

    costs = []
    n = len(theta)
    m = len(X)  
    Xt = np.transpose(X)
    grad = alpha*(1/m)

    for i in range(0,iteracoes):
        J = np.dot(X, theta)
        J = np.subtract(J,Y)
        cost = np.sum(np.square(J))/(2*m)
        new_theta = np.zeros(n)
        new_theta = np.subtract(theta,np.multiply(grad,np.dot(Xt, J)))
        theta = new_theta
        costs.append(cost)

def replace_dummies(train_features):
    train_features.cut[train_features['cut'] == 'Fair'] = 1
    train_features.cut[train_features['cut'] == 'Good'] = 2
    train_features.cut[train_features['cut'] == 'Very Good'] = 3
    train_features.cut[train_features['cut'] == 'Premium'] = 4
    train_features.cut[train_features['cut'] == 'Ideal'] = 5
    train_features.clarity[train_features['clarity'] == 'I1'] = 1
    train_features.clarity[train_features['clarity'] == 'SI2'] = 2
    train_features.clarity[train_features['clarity'] == 'SI1'] = 3
    train_features.clarity[train_features['clarity'] == 'VS2'] = 4
    train_features.clarity[train_features['clarity'] == 'VS1'] = 5
    train_features.clarity[train_features['clarity'] == 'VVS2'] = 6
    train_features.clarity[train_features['clarity'] == 'VVS1'] = 7
    train_features.clarity[train_features['clarity'] == 'IF'] = 8
    train_features.color[train_features['color'] == 'D'] = 7
    train_features.color[train_features['color'] == 'E'] = 6
    train_features.color[train_features['color'] == 'F'] = 5
    train_features.color[train_features['color'] == 'G'] = 4
    train_features.color[train_features['color'] == 'H'] = 3
    train_features.color[train_features['color'] == 'I'] = 2
    train_features.color[train_features['color'] == 'J'] = 1

def clean_data(dataset):
    dataset_labels = dataset["price"]
    dataset_features = dataset.drop(['Unnamed: 0', 'price'], axis=1)
    replace_dummies(dataset_features)
    #features
    train_features = dataset_features[0:36679]
    valid_features = dataset_features[36679:45849]
    #adicionando o X0 = 1
    x0 = np.ones((len(train_features),1))
    train_features = np.append(x0, train_features, axis=1)
    x0 = np.ones((len(valid_features),1))
    valid_features = np.append(x0, valid_features, axis=1)
    #targets
    train_labels = np.array(dataset_labels[0:36679])
    valid_labels = np.array(dataset_labels[36679:45849])

    #normalize_features(train_features)
    #normalize_features(valid_features)
    return train_features, valid_features, train_labels, valid_labels

def shape_csv(name):
    file = pd.read_csv(name,header=None)    
    file = np.array(file)
    file = file.astype(np.float)
    return file	
def main():
    dataset = pd.read_csv('diamonds.csv')
    train_features, valid_features, train_labels, valid_labels = clean_data(dataset)
    np.savetxt("train_features.csv",train_features,delimiter=",")
    np.savetxt("train_labels.csv",train_labels,delimiter=",")
    np.savetxt("valid_features.csv",valid_features,delimiter=",")
    np.savetxt("valid_labels.csv",valid_labels,delimiter=",")

    # Treino
    #theta = np.ones(10)
    iteracoes = 10000
    alpha = 0.0002
    prog=[]
    prog.append("./linearRegressionFlex")
    prog.append("-a="+str(alpha))
    prog.append("-i="+str(iteracoes))
    prog.append("-dvl=1")

    call(prog)

    #train_lr(theta, train_features, train_labels, iterations, alpha)

    costs = shape_csv('costs.csv')
    theta = shape_csv('theta.csv')
    predictions = shape_csv('predictions.csv')
    dash = '-' * 40

    for i in range(0, iteracoes, iteracoes//10):  
        print('Iteracao: {:<10d} Custo: {:<10f}'.format(i,costs[0][i]))

    print(dash)

    for i in range(0, len(theta)):
        print('Theta {:<1d} = {:<10f}'.format(i, theta[0][i])) 

    plt.plot(range(0,iteracoes),costs[0])
    plt.ylabel('Custo')
    plt.xlabel('Iterações')
    plt.show()

if __name__ == "__main__":
    main()




