import numpy as np
from numpy import linalg
import matplotlib
import pandas as pd
import warnings
from subprocess import call
def shape_csv(name):
    file = pd.read_csv(name,header=None)    
    file = np.array(file)
    file = file.astype(np.float)
    return file	

costs = shape_csv('costs.csv')
for i in costs[0]:
    if np.isnan(i):
        print("aaaa")
