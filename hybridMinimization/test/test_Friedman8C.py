########################################
# Benchmark function AND range
########################################
import numpy as np
#import importlib
#FUNCTION_FILE_PY = sys.argv[1]
#print('Reading function python file (need to be "function.py" as a function file): ',sys.argv)
#importlib.import_module(FUNCTION_FILE_PY)
global EXPERIMENT_NAME
EXPERIMENT_NAME = 'Friedman8C'
##Benchmark function (We want to maximize this function when MAXIMIZE_OBJECTIVE=True)
def objective(continuous_part,categorical_part):
    res = 0
    if categorical_part[0] == 0:
        res = res + 10*np.sin(np.pi*continuous_part[0]*continuous_part[1]) #max = 10
    res = res + 20*( continuous_part[2]-0.5 )**2 #max = 5
    if categorical_part[2] == 0:
        res = res + 10*continuous_part[3]
    if categorical_part[2] == 1:
        res = res - 10*continuous_part[3]
    if categorical_part[2] == 2:
        res = res + 5*continuous_part[3]
    #max so far = 10+5+10
    res = res + 5*continuous_part[4]
    #max so far = 30
    return res
#   
global categorical_list 
categorical_list = [[0,1,2],[0,1,2,3,4],[0,1,2],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1],[0,1]]
global continuous_list
continuous_list = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
global MAXIMIZE_OBJECTIVE
MAXIMIZE_OBJECTIVE = True
global NUM_OF_REPEATS
NUM_OF_REPEATS = 20
global evaluationBudget
evaluationBudget = 100 #number of evaluations 
global hybridStrategy
hybridStrategy = 'UCTS'
global methodList 
methodList = ['hybridM'] 
