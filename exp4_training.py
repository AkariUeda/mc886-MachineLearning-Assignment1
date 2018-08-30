import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
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
    iteracoes = int(sys.argv[1])
    alpha =float(sys.argv[2])
    gradient = sys.argv[3]
    
    #Plot settings
    matplotlib.style.use('seaborn')

    fig = plt.figure()
    train_plot = fig.add_subplot(2,1,1)
    train_plot.set_ylabel('Training cost')
    valid_plot = fig.add_subplot(2,1,2)
    valid_plot.set_ylabel('Validation cost')
    valid_plot.set_xlabel('Iterations')

    cores = ['tab:blue', 'tab:orange']

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
    #timestamps = shape_csv('times.csv')
    #Plotting
    if np.isfinite(costs[0]).all(): 
        train_plot.plot(range(0,iteracoes), costs[0], cores[0], label=str(alpha), linestyle='-')
    if np.isfinite(predictions[0]).all(): 
        valid_plot.plot(range(0,iteracoes), predictions[0], cores[1], label=str(alpha), linestyle='-')

    train_plot.legend()
    valid_plot.legend()
    fig.show()
    fig.savefig('training_'+str(alpha)+"_"+gradient+"_"+str(iteracoes)+'.png')

if __name__ == "__main__":
    main()




