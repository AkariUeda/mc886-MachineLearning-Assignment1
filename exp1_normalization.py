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
    iteracoes = 100
    alpha = [0.2, 0.002, 0.0002]
   
    #Plot settings
    matplotlib.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylabel('Custo')
    ax.set_xlabel('Iterações')
    cores = ['C0', 'C1', 'C2']

    for normalized in range(0, 2):
        train_features, valid_features, train_labels, valid_labels = get_data(normalized)

        for a in range(0,len(alpha)):
            prog=[]
            prog.append("./linearRegressionFlex")
            prog.append("-a="+str(alpha[a]))
            prog.append("-i="+str(iteracoes))
            prog.append("-dvl=0")
            prog.append("-async=1")
            print(prog)
            #prog.append("-vr=0")
            #Executes the call for C code
            call(prog)

            #train_lr(theta, train_features, train_labels, iterations, alpha)
            costs = shape_csv('costs.csv')
            theta = shape_csv('theta.csv')
            #predictions = shape_csv('predictions.csv')

            #Plotting
            if not np.isfinite(costs[0]).all(): 
                continue
            if(normalized):
                ax.plot(range(0,iteracoes), costs[0], cores[a], label='Norm ' + str(alpha[a]), linestyle='--')
            else:
                ax.plot(range(0,iteracoes), costs[0], cores[a], label=str(alpha[a]), linestyle='-')

    ax.legend()
    plt.savefig('iters_'+str(iteracoes)+'.png')
    fig.show()

if __name__ == "__main__":
    main()




