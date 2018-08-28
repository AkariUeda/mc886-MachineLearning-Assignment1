import numpy as np
from numpy import linalg
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from get_dataset import get_data
from subprocess import call

warnings.filterwarnings('ignore')


def shape_csv(name):
    file = pd.read_csv(name,header=None)    
    file = np.array(file)
    file = file.astype(np.float)
    return file	

def main():
    train_features, valid_features, train_labels, valid_labels = get_data()

    # Adjusting training parameters
    #theta = np.ones(10)
    iteracoes = 10000
    alpha = 0.2
    prog=[]
    prog.append("./linearRegressionFlex")
    prog.append("-a="+str(alpha))
    prog.append("-i="+str(iteracoes))
    prog.append("-dvl=1")
    prog.append("-async=1")


    #Executes the call for C code
    call(prog)

    #train_lr(theta, train_features, train_labels, iterations, alpha)

    costs = shape_csv('costs.csv')
    theta = shape_csv('theta.csv')
    predictions = shape_csv('predictions.csv')
    dash = '-' * 40

    for i in range(0, iteracoes, iteracoes//10):  
        print('Iteracao: {:<10d} Custo: {:<10f}'.format(i,costs[0][i]))

    print(dash)

    #for i in range(0, len(theta)):
    #    print('Theta {:<1d} = {:<10f}'.format(i, theta[0][i])) 

    plt.plot(range(0,iteracoes),costs[0])
    plt.ylabel('Custo')
    plt.xlabel('Iterações')
    plt.show()

if __name__ == "__main__":
    main()




