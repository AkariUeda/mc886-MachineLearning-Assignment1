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

    fig_pred = plt.figure()
    pred = fig_pred.add_subplot(1,1,1)
    pred.set_ylabel('Price')
    pred.set_xlabel('Examples')

    cores = ['tab:blue', 'tab:orange']

    train_features = shape_csv('train_features.csv')
    train_labels = shape_csv('train_labels.csv')
    valid_features = shape_csv('valid_features.csv')
    valid_labels = shape_csv('valid_labels.csv')
    print(train_features.shape)

    print(train_features.shape)
    #Drop feature x, y and z (columns 7, 8 and 9)
    a = train_features[:,[1,2,3,4]]
    a = np.power(a,-1,dtype=float)
    a = a.sum(1)
    a = np.power(a,-1,dtype=float)
    a = 9*a
    a = a.reshape(-1,1)
    b = valid_features[:,[1,2,3,4]]   
    b = np.append(b,np.square(valid_features[:,[1,2,3,4]]),axis=1) 
    b = np.power(b,-1,dtype=float)
    b = b.sum(1)
    b = np.power(b,-1,dtype=float)
    b = 9*b
    b = b.reshape(-1,1)
    valid_features = np.append(valid_features,b,axis=1)
    train_features = np.append(train_features,a,axis=1)


    print(train_features.shape)


    np.savetxt("dropped_valid_features.csv", valid_features, delimiter=",")
    np.savetxt("dropped_train_features.csv", train_features, delimiter=",")
    np.savetxt("dropped_valid_labels.csv", valid_labels, delimiter=",")
    np.savetxt("dropped_train_labels.csv", train_labels, delimiter=",")
    #Setttings training parameters
    prog=[]
    prog.append("./linearRegressionFlex")
    prog.append("-a="+str(alpha))
    prog.append("-i="+str(iteracoes))
    prog.append("-dvl=1")
    prog.append("-async=1")
    prog.append("-vr=1")
    prog.append("-rd=1")
    prog.append("-f=11")
    prog.append("-tf=dropped_train_features.csv")
    prog.append("-tl=dropped_train_labels.csv")    
    prog.append("-vf=dropped_valid_features.csv")
    prog.append("-vl=dropped_valid_labels.csv")
    if(gradient == 'batch'):
        prog.append("-sgd=0")
    elif(gradient == 'sgd'):
        prog.append("-sgd=1")
    elif(gradient == 'minibatch'):
        prog.append("-mb=1")

    #Executes the call for C code
    call(prog)
    #train_lr(theta, train_features, train_labels, iterations, alpha)
    costs = shape_csv('costs.csv')
    theta = shape_csv('theta.csv')
    predictions = shape_csv('predictCosts.csv')

    #timestamps = shape_csv('times.csv')
    #Plotting
    if np.isfinite(costs[0]).all(): 
        train_plot.plot(range(0,len(costs[0])), costs[0], cores[0], label=str(alpha), linestyle='-')
    if np.isfinite(predictions[0]).all(): 
        valid_plot.plot(range(0,len(predictions[0])), predictions[0], cores[1], label=str(alpha), linestyle='-')

    h = np.dot(valid_features, theta[0])

    valid_labels, H = zip(*sorted(zip(valid_labels, h)))

    pred.plot(range(0, len(H)),H, 'b.', label="Predicted")
    pred.plot(range(0, len(valid_labels)), valid_labels, 'r.', label="Target")

    print(len(valid_labels))

    pred.legend()
    train_plot.legend()
    valid_plot.legend()
    fig_pred.show()
    fig_pred.savefig('drop_prediction_'+str(alpha)+"_"+gradient+"_"+str(iteracoes)+'.png')
    fig.show()
    fig.savefig('drop_training_'+str(alpha)+"_"+gradient+"_"+str(iteracoes)+'.png')

if __name__ == "__main__":
    main()




