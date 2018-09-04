import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys

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
    gradient = sys.argv[1]
    time = sys.argv[2]
    alpha = [0.2, 0.02, 0.002, 0.0002]
    #Plot settings
    matplotlib.style.use('seaborn')
    #plt.yscale("log")

    fig = plt.figure()
    train_plot = fig.add_subplot(2,1,1)
    train_plot.set_ylabel('Training cost')
    valid_plot = fig.add_subplot(2,1,2)

    valid_plot.set_ylabel('Validation cost')
    valid_plot.set_xlabel('Time(s)')
    cores = ['tab:blue',  'tab:green', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown']
   	
    if(gradient == "batch"):
        g = "-sgd=0"
    elif(gradient == "sgd"):
        g="-sgd=1"
    else:
        g="-mb=1"

    train_features = shape_csv('train_features.csv')
    train_labels = shape_csv('train_labels.csv')
    valid_features = shape_csv('valid_features.csv')
    valid_labels = shape_csv('valid_labels.csv')

    for a in range(0,len(alpha)):
        prog=[]
        prog.append("./linearRegressionFlex")
        prog.append("-a="+str(alpha[a]))
        prog.append("-time="+str(time))
        prog.append("-dvl=1")
        prog.append(g)
        prog.append("-async=1")
        prog.append("-vr=0")

        #Executes the call for C code
        #print(prog)            
        call(prog)
        
        #train_lr(theta, train_features, train_labels, iterations, alpha)
        theta = shape_csv('theta.csv')
        #print(theta)
        if not np.isfinite(theta[0]).all():
            continue 
        costs = shape_csv('costs.csv')
        predictions = shape_csv('predictCosts.csv')
        timestamps = shape_csv('times.csv')

        #Plotting
        train_plot.plot(timestamps[0]/1000, costs[0], cores[a], label=gradient+ " " +str(alpha[a]), linestyle='-')
        valid_plot.plot(timestamps[0]/1000, predictions[0], cores[a], label=gradient+ " " +str(alpha[a]), linestyle='-')
        print("Learning rate: " + str(alpha[a]))
        print("   Training cost: " + str(costs[0, len(costs[0])-1]))
        print("   Validation cost: " +str(predictions[0, len(predictions[0])-1]))

    train_plot.legend()
    valid_plot.legend()
    fig.show()
    fig.savefig('gd_'+ gradient +str(time)+'.png')

if __name__ == "__main__":
    main()




