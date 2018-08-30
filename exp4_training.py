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
    iteracoes = 1000
    alpha = 0.002
   
    #Plot settings
    matplotlib.style.use('seaborn')
    fig_train, train_plot = plt.subplots(figsize=(10, 5))
    train_plot.set_ylabel('Cost')
    train_plot.set_xlabel('Iterations')

    fig_valid, valid_plot = plt.subplots(figsize=(10, 5))
    valid_plot.set_ylabel('Cost')
    valid_plot.set_xlabel('Iterations')
    cores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    train_features, valid_features, train_labels, valid_labels = get_data(1)


    prog=[]
    prog.append("./linearRegressionFlex")
    prog.append("-a="+str(alpha))
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
    if np.isfinite(costs[0]).all(): 
        train_plot.plot(range(0,iteracoes)/1000, costs[0], cores[0], label=str(alpha), linestyle='-')
    if np.isfinite(predictions[0]).all(): 
        valid_plot.plot(range(0,iteracoes)/1000, predictions[0], cores[0], label=str(alpha), linestyle='-')

    train_plot.legend()
    valid_plot.legend()
    fig_train.show()
    fig_train.savefig('training_'+str(iteracoes)+'.png')
    fig_valid.show()
    fig_valid.savefig('validation_'+str(iteracoes)+'.png')

    fig_train, train_plot = plt.subplots(figsize=(10, 5))
    train_plot.set_ylabel('Cost')
    train_plot.set_xlabel('Iterations')

    fig_valid, valid_plot = plt.subplots(figsize=(10, 5))
    valid_plot.set_ylabel('Cost')
    valid_plot.set_xlabel('Iterations')
if __name__ == "__main__":
    main()




