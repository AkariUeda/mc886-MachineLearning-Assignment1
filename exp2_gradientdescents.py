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
    time = 5
    alpha = [0.2, 0.02, 0.002, 0.0002]
   
    #Plot settings
    matplotlib.style.use('seaborn')
    fig_train, train_plot = plt.subplots(figsize=(10, 5))
    train_plot.set_ylabel('Cost')
    train_plot.set_xlabel('Time')

    fig_valid, valid_plot = plt.subplots(figsize=(10, 5))
    valid_plot.set_ylabel('Cost')
    valid_plot.set_xlabel('Time')
    cores = ['tab:blue',  'tab:green', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown']
    #"-sgd=1",    
    gradients = ["-sgd=0", "-mb=1"]

    train_features, valid_features, train_labels, valid_labels = get_data(1)
    for g in range(0,len(gradients)):
        for a in range(0,len(alpha)):
            prog=[]
            prog.append("./linearRegressionFlex")
            prog.append("-a="+str(alpha[a]))
            #prog.append("-time="+str(time))
            
            prog.append("-dvl=1")
            prog.append(gradients[g])
            prog.append("-async=1")
            prog.append("-vr=0")

            #Executes the call for C code
            call(prog)

            #train_lr(theta, train_features, train_labels, iterations, alpha)
            costs = shape_csv('costs.csv')
            theta = shape_csv('theta.csv')
            predictions = shape_csv('predictCosts.csv')
            timestamps = shape_csv('times.csv')

            #Plotting
            if not np.isfinite(costs[0]).all(): 
                continue
            elif g==0:
                train_plot.plot(timestamps[0,100:], costs[0, 100:], cores[a], label='Batch ' + str(alpha[a]), linestyle='--')
            #elif g==1:
            #    train_plot.plot(timestamps[0,100:], costs[0, 100:], cores[a], label='SGD ' + str(alpha[a]), linestyle='-')
            elif g==1:
                train_plot.plot(timestamps[0,100:], costs[0, 100:], cores[a], label='MiniBatch ' + str(alpha[a]), linestyle=':')
               
            if not np.isfinite(predictions[0]).all(): 
                continue
            elif g==0:
                valid_plot.plot(timestamps[0,100:], predictions[0, 100:], cores[a], label='Batch ' + str(alpha[a]), linestyle='--')
            #elif g==1:
            #    valid_plot.plot(timestamps[0,100:], predictions[0, 100:], cores[a], label='SGD ' + str(alpha[a]), linestyle='-')
            elif g==1:
                valid_plot.plot(timestamps[0,100:], predictions[0, 100:], cores[a], label='MiniBatch ' + str(alpha[a]), linestyle=':')

    train_plot.legend()
    valid_plot.legend()
    fig_train.show()
    fig_train.savefig('train_gd_'+str(time)+'.png')
    fig_valid.show()
    fig_valid.savefig('valid_gd_'+str(time)+'.png')
if __name__ == "__main__":
    main()




