########################################
# Benchmark function AND range
########################################
import numpy as np
from syntheticFunctions import *
#import importlib
#FUNCTION_FILE_PY = sys.argv[1]
#print('Reading function python file (need to be "function.py" as a function file): ',sys.argv)
#importlib.import_module(FUNCTION_FILE_PY)
global EXPERIMENT_NAME
EXPERIMENT_NAME = 'rosenbrock2'
##Benchmark function (We want to maximize this function when MAXIMIZE_OBJECTIVE=True)
def objective(continuous_part,categorical_part):
    categorical_part = np.array(categorical_part).reshape(1,-1)
    continuous_part = np.array(continuous_part).reshape(1,-1)
    z = np.hstack((continuous_part,categorical_part))
    #print(categorical_part)
    res = 0
    for i in range(z.shape[1]-1):
        if i == 0: continue
        xi0 = z[:,i]
        xi1 = z[:,(i+1)]
        #print(xi0,xi1)
        res = res + 100*(xi1-xi0**2)**2 + (1-xi0)**2
    print('res =', res)
    return -1.*res[0]/100000
#   
global categorical_list 
categorical_list = [[-5,-4,-3,-2,-1,0,1,2,3,4,5]]*4
global continuous_list
continuous_list = [[-5,5],[-5,5],[-5,5],[-5,5]]
global MAXIMIZE_OBJECTIVE
MAXIMIZE_OBJECTIVE = True
global NUM_OF_REPEATS
NUM_OF_REPEATS = 20
global evaluationBudget
evaluationBudget = 100 #number of evaluations 
global hybridStrategy
hybridStrategy = 'UCTS'
global methodList 
methodList = [] 
