import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

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

    # Adjusting training parameters
    iteracoes = 20000
    alpha = [0.2, 0.01, 0.02, 0.002, 0.0002, 0.00002]
   
    #Plot settings
    matplotlib.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylabel('Custo')
    ax.set_xlabel('Iterações')
    cores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    train_features, valid_features, train_labels, valid_labels = get_data(1)

    for a in range(0,len(alpha)):
        prog=[]
        prog.append("./linearRegressionFlex")
        prog.append("-a="+str(alpha[a]))
        prog.append("-i="+str(iteracoes))
        prog.append("-dvl=0")
        prog.append("-async=1")
        prog.append("-vr=0")

        #Executes the call for C code
        call(prog)
        print(prog)
        #train_lr(theta, train_features, train_labels, iterations, alpha)
        costs = shape_csv('costs.csv')
        theta = shape_csv('theta.csv')
        #predictions = shape_csv('predictions.csv')

        #Plotting
        if not np.isfinite(costs[0]).all(): 
            continue
        else:
            ax.plot(range(1000,iteracoes), costs[0, 1000:], cores[a], label='Batch ' + str(alpha[a]), linestyle='-')
       
    ax.legend()
    plt.savefig('lr_'+str(iteracoes)+'.png')
    fig.show()

if __name__ == "__main__":
    main()




