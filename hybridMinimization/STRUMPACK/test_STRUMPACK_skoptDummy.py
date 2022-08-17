########################################
# Benchmark application-defined function AND range
########################################
import numpy as np
import warnings, time

from strumpack_MLA_Poisson3d_simple import objectives
#default
point0 = {'nodes':1,'cores':2,'gridsize':20,'sp_reordering_method':'metis','sp_compression':'hss','sp_compression_min_sep_size':2,'sp_compression_leaf_size':5,'sp_compression_rel_tol':-6}
def f_truth_STRUMPACK(X=point0):
    #point1 = {'nodes':1,'cores':2,'gridsize':20,'sp_reordering_method':'metis','sp_compression':'hss','sp_compression_min_sep_size':2,'sp_compression_leaf_size':5,'sp_compression_rel_tol':-6}
    result1 = objectives(X)
    #print('Success!')
    #print(result1)
    return result1[0]
#print(f_truth_STRUMPACK(point0))

           
global EXPERIMENT_NAME
EXPERIMENT_NAME = 'STRUMPACK'
##Benchmark function (We want to maximize this function when MAXIMIZE_OBJECTIVE=True)
def objective(continuous_part,categorical_part):
    categorical_part = np.array(categorical_part).reshape(1,-1)
    continuous_part = np.array(continuous_part).reshape(1,-1)
    z = np.hstack((continuous_part,categorical_part))
    print(continuous_part,categorical_part,z)
    res = 0
    n_model_cat = categorical_part[0,1]
    
    #Categorical 1
    #sp_reordering_method   = Categoricalnorm (['metis','parmetis','geometric'], transform="onehot", name="sp_reordering_method")
    sp_reordering_method = ''
    if categorical_part[0,0] == 0:
        sp_reordering_method = 'metis'
    if categorical_part[0,0] == 1:
        sp_reordering_method = 'parmetis'
    if categorical_part[0,0] == 2:
        sp_reordering_method = 'geometric'   
        
    #Categorical 2
    #sp_compression   = Categoricalnorm (['hss','hodbf','blr'], transform="onehot", name="sp_compression")
    sp_compression = ''
    if categorical_part[0,1] == 0:
        sp_compression = 'hss'
    if categorical_part[0,1] == 1:
        sp_compression = 'hodbf'
    if categorical_part[0,1] == 2:
        sp_compression = 'blr'   
        
    #Continuous 1 or Categorical 3
    #sp_compression_min_sep_size     = Integer     (2, 5, transform="normalize", name="sp_compression_min_sep_size")
    sp_compression_min_sep_size = np.floor(continuous_part[0,0]).astype(int)
    sp_compression_min_sep_size = 0
    if categorical_part[0,2] == 0:
        sp_compression_min_sep_size = 2
    if categorical_part[0,2] == 1:
        sp_compression_min_sep_size = 3
    if categorical_part[0,2] == 2:
        sp_compression_min_sep_size = 4
    if categorical_part[0,2] == 3:
        sp_compression_min_sep_size = 5
   
    #Continuous 1
    #sp_compression_leaf_size     = Integer     (5, 9, transform="normalize", name="sp_compression_leaf_size")
    sp_compression_leaf_size = np.floor(continuous_part[0,0]).astype(int)
   
    #Continuous 2
    #sp_compression_rel_tol     = Integer(-6, -1, transform="normalize", name="sp_compression_rel_tol")
    sp_compression_rel_tol = np.floor(continuous_part[0,1]).astype(int)
   
    point1 = {'nodes':8,'cores':8,'gridsize':100,\
              'sp_reordering_method':sp_reordering_method,\
              'sp_compression':sp_compression,\
              'sp_compression_min_sep_size':sp_compression_min_sep_size,\
              'sp_compression_leaf_size':sp_compression_leaf_size,\
              'sp_compression_rel_tol':sp_compression_rel_tol}
    #**objectives**, not objective without an s.
    res = objectives(point1)
    res = res[0]
    res = float(res)
    print('res =', res)
    return -1.*np.array(res) #We want to minimize the running time, or equivalently, maximizing its negative.
#   
#print(objective(categorical_part=[0,0,0],continuous_part=[5,-3]))
#
global categorical_list 
categorical_list = [[0,1,2],[0,1,2],[0,1,2,3]]
global continuous_list
continuous_list = [[5,9],[-6,-1]]                   
global MAXIMIZE_OBJECTIVE
MAXIMIZE_OBJECTIVE = True
global NUM_OF_REPEATS
NUM_OF_REPEATS = 15
global evaluationBudget
evaluationBudget = 100 #number of evaluations 
global hybridStrategy
hybridStrategy = 'UCTS'
global methodList 
#methodList = ['hybridM'] 
methodList = ['skoptDummy']#,'roundrobinMAB','randomMAB','hybridM','skoptForest','skoptGP','skoptDummy']
global categorical_model
from treeClass import setupTree
categorical_model = setupTree(categorical_list=categorical_list,\
                          policy_list=[hybridStrategy]*len(categorical_list),update_list=[hybridStrategy]*len(categorical_list),\
                          exploration_probability=0.10,\
                          print_tree=True)
#for k in [4,5,6,7,10,11,15]:
#    categorical_model.searchKey(k)[0].removeMyself() 
categorical_model.printTree()     
