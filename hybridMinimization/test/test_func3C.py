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
EXPERIMENT_NAME = 'func3C'
##Benchmark function (We want to maximize this function when MAXIMIZE_OBJECTIVE=True)
def objective(continuous_part,categorical_part):
        res = 0
        ht_list = categorical_part
        X = np.asarray(continuous_part)
        res = func3C(ht_list, X)
        res = res[0]
        return res[0]
#   
global categorical_list 
categorical_list = [[0,1,2],[0,1,2,3,4],[0,1,2,3]]
global continuous_list
continuous_list = [[-1,1],[-1,1]]
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
