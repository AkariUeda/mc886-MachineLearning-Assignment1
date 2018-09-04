import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import warnings
from subprocess import call

warnings.filterwarnings('ignore')

def shape_csv(name):
    file = pd.read_csv(name,header=None)    
    file = np.array(file)
    file = file.astype(np.float)
    return file    

def main():

    # Adjusting training parameters
    iteracoes = 10000
    alpha = [0.2, 0.1, 0.02, 0.002, 0.0002]
   
    #Plot settings
    matplotlib.style.use('seaborn')
    fig_train, train_plot = plt.subplots(figsize=(10, 5))
    train_plot.set_ylabel('Cost')
    train_plot.set_xlabel('Iterations')

    fig_valid, valid_plot = plt.subplots(figsize=(10, 5))
    valid_plot.set_ylabel('Cost')
    valid_plot.set_xlabel('Iterations')
    cores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    train_features = shape_csv('train_features.csv')
    train_labels = shape_csv('train_labels.csv')
    valid_features = shape_csv('valid_features.csv')
    valid_labels = shape_csv('valid_labels.csv')

    for a in range(0,len(alpha)):
        prog=[]
        prog.append("./linearRegressionFlex")
        prog.append("-a="+str(alpha[a]))
        prog.append("-i="+str(iteracoes))
        prog.append("-dvl=1")
        prog.append("-async=1")
        prog.append("-vr=1")
        prog.append("-rd=1")

        #Executes the call for C code
        call(prog)
        print(prog)
        #train_lr(theta, train_features, train_labels, iterations, alpha)
        costs = shape_csv('costs.csv')
        theta = shape_csv('theta.csv')
        predictions = shape_csv('predictCosts.csv')

        #Plotting
        if not np.isfinite(theta[0]).all(): 
            continue
        else:
            train_plot.plot(range(0,iteracoes), costs[0], cores[a], label=str(alpha[a]), linestyle='-')

       
        if not np.isfinite(theta[0]).all(): 
            continue
        else:
            valid_plot.plot(range(0,iteracoes), predictions[0], cores[a], label=str(alpha[a]), linestyle='-')

    train_plot.legend()
    valid_plot.legend()
    fig_train.show()
    fig_train.savefig('train_lr_'+str(iteracoes)+'.png')
    fig_valid.show()
    fig_valid.savefig('valid_lr_'+str(iteracoes)+'.png')
if __name__ == "__main__":
    main()




