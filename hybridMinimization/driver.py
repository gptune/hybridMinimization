#!/usr/bin/env python
# coding: utf-8

# # Benchmarking with Multiple Mixed-variable Surrogate-based Optimization
# ### Hengrui Luo 2020-11-26
import sys
#print(sys.path)
import numpy as np
import random, string, importlib
#This stores the trained model and the relevant dataset into a .pkl object for later loadings               
import dill as pkl
  
DESTINATION = '/media/hrluo/WORK/LBNLhrluo/minimalMCTS/'

#If provided, just append new runs, otherwise create new RUNNING_STR as a stamp.
RUNNING_STR = ''#'J2GLXVX8M28RV36VUGIWEHRW'
if len(RUNNING_STR)<=3:
    RUNNING_STR = ''.join(random.choices(string.ascii_uppercase + string.digits, k=24))
    RUNNING_STR = 'bigbenchmark_'+RUNNING_STR
WORKING_DIR = RUNNING_STR+'/' 

import os
# If folder doesn't exist, then create it.
if not os.path.isdir(WORKING_DIR):
    os.makedirs(WORKING_DIR)
    print("created folder : ", WORKING_DIR)
else:
    print(WORKING_DIR, "folder already exists.")

#Which methods we are going to execute?
methodList = []
if len(methodList)<=0:
    methodList = ['CoCaBO','TPE','SMAC','EXP3BO','roundrobinMAB','randomMAB','hybridM','skoptForest','skoptGP','skoptDummy']

########################################
# Benchmark function AND range
########################################
#
import importlib
FUNCTION_FILE_PY = sys.argv[1]
print('Reading function python file (need to be "function" as a function file, without .py extension): ',sys.argv)
my_module = importlib.__import__(FUNCTION_FILE_PY)
EXPERIMENT_NAME = my_module.EXPERIMENT_NAME
categorical_list = my_module.categorical_list
continuous_list = my_module.continuous_list
MAXIMIZE_OBJECTIVE = my_module.MAXIMIZE_OBJECTIVE
NUM_OF_REPEATS = my_module.NUM_OF_REPEATS
evaluationBudget = my_module.evaluationBudget
objective = my_module.objective
if len(my_module.methodList) > 0:
    methodList = my_module.methodList

#Dump the information into a json file for reference and visualization script.                       
STAMP = WORKING_DIR+EXPERIMENT_NAME
import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
DATE_TIME_STR = now.strftime("%Y-%m-%d %H:%M:%S")
print(DATE_TIME_STR)
import json
info_list = {'destination':DESTINATION,\
             'running_str':RUNNING_STR,\
             'num_of_repeats':NUM_OF_REPEATS,\
             'maximize_objective':MAXIMIZE_OBJECTIVE,\
             'experiment_name':EXPERIMENT_NAME,\
             'function_file_py':FUNCTION_FILE_PY,\
             'working_dir':WORKING_DIR,\
             'date_time_str':DATE_TIME_STR,\
             'evaluation_budget':evaluationBudget,\
             'method_list':methodList
             }
filename = 'json/'+RUNNING_STR+'.json'
with open(filename, 'w') as f:
    f.write(json.dumps(info_list))

from universalConfigConversion import *
CoCaBO_space = list_to_config(categorical_list,continuous_list,target_format='CoCaBO')
CoCaBO_catCount = [len(k) for k in categorical_list]
print(CoCaBO_catCount)

TPE_space = list_to_config(categorical_list,continuous_list,target_format='TPE')
SMAC_space = list_to_config(categorical_list,continuous_list,target_format='SMAC')
EXP3BO_space = list_to_config(categorical_list,continuous_list,target_format='EXP3BO')
categoriesN = 1
for j in categorical_list:
    categoriesN = categoriesN*len(j)
print(categoriesN)
skopt_space = list_to_config(categorical_list,continuous_list,target_format='skopt')
    
########################################
# REPEAT LOOPS
########################################   
#The repeated experiments, considering the cost of time, NUM_OF_REPEATS should not exceed 10 if the script has not been tested.
if len(sys.argv)>=3:
    STARTING_SEED_NUM = int(sys.argv[2])
else:
    STARTING_SEED_NUM = 0
for randomSeed in range(STARTING_SEED_NUM,NUM_OF_REPEATS):
    print('\n\n\n\n\n\n\n\n\n\n')
    print('total repeats:',NUM_OF_REPEATS,' currently, randomSeed = ',randomSeed,' below are this loop.')
    print('\n\n\n\n\n\n\n\n\n\n')
    
    #Each loop has a different seed
    RNG = np.random.RandomState(randomSeed)
    
    #Shared budget settings.
    trialsN = 1      # no of times to repeat the experiment
    initialsN = 1    # no of initial samples for the experiment
    budget = evaluationBudget     # budget for bayesian optimisation

    ########################################
    # HybridMinimization (DEFAULT: min)
    ########################################
    #
    if 'hybridM' in methodList or 'hybridM2' in methodList or 'hybridM4' in methodList:
        
        HM_path=DESTINATION
        sys.path.insert(0, HM_path)
        #import hybridMinimization
        #from hybridMinimization import *
        from hybridMinimization import *
        
        preset_tree_exists = 'categorical_model' in locals() or 'categorical_model' in globals()
        if preset_tree_exists:
            cm = categorical_model
        else:
            cm = None
        #
        def f_truth(X):
            categorical_part = np.asarray(X[0:len(categorical_list)])
            continuous_part = np.asarray(X[len(categorical_list):])
            #print((continuous_part,categorical_part))
            Y = objective(continuous_part=continuous_part,categorical_part=categorical_part)
            return Y
        #
        np.random.seed(randomSeed)
        X0_cat = [np.random.choice(i,size=initialsN)[0] for i in categorical_list]
        X0_con = [np.random.uniform(low=j[0],high=j[1],size=initialsN)[0] for j in continuous_list]
        X0 = np.asarray(X0_cat+X0_con)
        Y0 = f_truth(X0).reshape(1,-1)
        X0 = X0.reshape(1,-1)
        #
        if my_module.hybridStrategy is not None:
            hybridStrategy = my_module.hybridStrategy
        else:
            hybridStrategy = 'UCTS'
        attempt_setup = []
        if 'hybridM' in methodList:attempt_setup.append(evaluationBudget)
        if 'hybridM2' in methodList:attempt_setup.append(int(evaluationBudget/2))
        if 'hybridM4' in methodList:attempt_setup.append(int(evaluationBudget/4))
        my_selection='custom'
        for n_find_tree_iter in attempt_setup:
            h1_y,h1_x,h1_root,h1_model,h1_model_history = hybridMinimization(fn=f_truth,\
                                                   selection_criterion = my_selection,fix_model = -1,\
                                                   categorical_list=categorical_list,\
                                                   categorical_trained_model=cm,\
                                                   policy_list=[hybridStrategy]*len(categorical_list),update_list=[hybridStrategy]*len(categorical_list),exploration_probability=0.10,\
                                                   continuous_list=continuous_list,\
                                                   continuous_trained_model=None,\
                                                   observed_X = X0,\
                                                   observed_Y = Y0,\
                                                   n_find_tree=n_find_tree_iter,\
                                                   n_find_leaf=int(evaluationBudget/n_find_tree_iter),\
                                                   node_optimize='GP-EI',\
                                                   random_seed=randomSeed,minimize=np.logical_not(MAXIMIZE_OBJECTIVE),\
                                                   N_initialize_leaf=0)#or, to set N_initialize_leaf=initialsN to get at least one observation per leaf. 
            #h1_root.printTree()
            pkl_dict = {'categorical_model':h1_root,'continuous_model':h1_model,'train_X':h1_x,'train_Y':h1_y,'model_history':h1_model_history,'continuous_list':continuous_list,'categorical_list':categorical_list}
              
            print('=====HybridMinimization result=====')
            #print(list(h1_y.reshape(1,-1)))
            #print('>>>>>total evaluations: ',h1_model.num_data)
            if n_find_tree_iter == evaluationBudget:
                np.savetxt(STAMP+'_hybridM_'+str(randomSeed)+'_X.csv', h1_x[0:h1_x.shape[0],:], delimiter=",")
                np.savetxt(STAMP+'_hybridM_'+str(randomSeed)+'_Y.csv', h1_y[0:h1_x.shape[0]], delimiter=",")
                OBJ_NAME = STAMP+'_hybridM_'+str(randomSeed)+'.pkl' 
            elif n_find_tree_iter == int(evaluationBudget/2):
                np.savetxt(STAMP+'_hybridM2_'+str(randomSeed)+'_X.csv', h1_x[0:h1_x.shape[0],:], delimiter=",")
                np.savetxt(STAMP+'_hybridM2_'+str(randomSeed)+'_Y.csv', h1_y[0:h1_x.shape[0]], delimiter=",")
                OBJ_NAME = STAMP+'_hybridM2_'+str(randomSeed)+'.pkl' 
            elif n_find_tree_iter == int(evaluationBudget/4):
                np.savetxt(STAMP+'_hybridM4_'+str(randomSeed)+'_X.csv', h1_x[0:h1_x.shape[0],:], delimiter=",")
                np.savetxt(STAMP+'_hybridM4_'+str(randomSeed)+'_Y.csv', h1_y[0:h1_x.shape[0]], delimiter=",")
                OBJ_NAME = STAMP+'_hybridM4_'+str(randomSeed)+'.pkl' 
            with open(OBJ_NAME, "wb+") as file_model: 
              pkl.dump(pkl_dict, file_model)
        sys.path.remove(HM_path)

    ########################################
    # CoCaBO (DEFAULT: maximize)
    ########################################
    #
    if 'CoCaBO' in methodList:
    
        np.random.seed(randomSeed)
        CoCaBO_path=DESTINATION+'CoCaBO_code/'
        sys.path.insert(0, CoCaBO_path)
        print(sys.path)
        # 
        # =============================================================================
        #  CoCaBO Algorithms 
        # =============================================================================
        from methods.CoCaBO import CoCaBO
        from methods.BatchCoCaBO import BatchCoCaBO
        def CoCaBO_objective(ht, x):
            # ht is a categorical index
            # X is a continuous variable
            x = x.reshape(-1,len(continuous_list))
            y_list = []
            ht_coded = []
            for k in range(len(categorical_list)):
                ht_coded.append( categorical_list[k][ht[k]] )
            print('ht_coded',ht_coded)
            ht_coded = np.asarray(ht_coded)
            for k in range(x.shape[0]):  
                continuous_part = x[k,:]
                #[x[k,i] for i in range(x.shape[1])]
                categorical_part = ht_coded
                print(categorical_part,continuous_part)
                if MAXIMIZE_OBJECTIVE==False:
                    y_list.append( -1.*objective(continuous_part,categorical_part) )
                else:
                    y_list.append( 1.*objective(continuous_part,categorical_part) )
            y_list = np.asarray(y_list)
            return y_list.reshape(-1,1)


        # define saving path for saving the results
        saving_path = WORKING_DIR+'CoCaBO/'+''.join(random.choice(string.ascii_lowercase) for i in range(24))
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        myCoCaBO = CoCaBO(objfn=CoCaBO_objective, initN=initialsN, bounds=CoCaBO_space,
                       acq_type='LCB', C=CoCaBO_catCount,
                       kernel_mix = 0.5)
        #By default the mixture coefficient is 0.5
        #myCoCaBO.runTrials(evaluationBudget,NUM_OF_REPEATS, saving_path)#NUM_OF_REPEATS is the number of trials in the original code, meaning that the experiments should be repeated for that many times.
        myCoCaBO.saving_path = saving_path
        myCoCaBO.trial_num = 1
        myCoCaBO.runOptim(budget=evaluationBudget-initialsN, seed=randomSeed)

        print('=====CoCaBO result=====')
        #print(CoCaBO_result)
        #print('>>>>>total evaluations: ',len(CoCaBO_result))
        CoCaBO_best_x = myCoCaBO.data[0]
        CoCaBO_best_y = myCoCaBO.result[0]
        
        np.savetxt(STAMP+'_CoCaBO_'+str(randomSeed)+'_X.csv', np.asarray(CoCaBO_best_x), delimiter=",")
        if MAXIMIZE_OBJECTIVE==False:
            np.savetxt(STAMP+'_CoCaBO_'+str(randomSeed)+'_Y.csv', np.asarray(CoCaBO_best_y)*(-1.), delimiter=",")
        else:
            np.savetxt(STAMP+'_CoCaBO_'+str(randomSeed)+'_Y.csv', np.asarray(CoCaBO_best_y)*(1.), delimiter=",")
        sys.path.remove(CoCaBO_path)
    ########################################


    ########################################
    # TPE  (DEFAULT: min)
    ########################################
    #
    if 'TPE' in methodList:
        
        np.random.seed(randomSeed)
        from hyperopt import fmin, rand, tpe, anneal, hp, STATUS_OK, Trials

        def TPE_objective(x):
            h_counter = 0
            x_counter = 0
            continuous_part = []
            categorical_part = []
            for k in x.keys():
                if k[0]=='x':
                    continuous_part.append(x[k])
                    x_counter = x_counter +1
                else:
                    categorical_part.append(x[k])
                    h_counter = h_counter +1
            #print(len(categorical_part),categorical_part)
            if MAXIMIZE_OBJECTIVE==False:
                return {
                    'loss': objective(continuous_part,categorical_part),
                    'status': STATUS_OK,
                    'variable':x,
                    # -- store other results like this
                    }
            else:
                return {
                    'loss': -1.*objective(continuous_part,categorical_part),
                    'status': STATUS_OK,
                    'variable':x,
                    # -- store other results like this
                    }


        TPE_trials = Trials()
        TPE_best = fmin(TPE_objective,
            space=TPE_space,
            algo=tpe.suggest,
            max_evals=evaluationBudget,
            rstate=RNG,
            trials=TPE_trials)

        #Dump the results
        print('=====TPE result=====')
        #print(TPE_trials.trials)
        #print('>>>>>total evaluations: ',len(TPE_trials.trials))
        #print(TPE_objective(TPE_best)['loss'])
        TPE_history = []
        for j in range(len(TPE_trials.results)):
            TPE_history.append(TPE_trials.results[j]['loss'])
        #print(TPE_history,'\n',TPE_best)
        print('>>>>>total evaluations: ',len(TPE_history))

        TPE_x = np.zeros((evaluationBudget,0))
        for i, (k, v) in enumerate(TPE_trials.vals.items()):
            #print(i, k, len(v))
            #print(np.asarray(v).shape)
            TPE_x = np.hstack(( TPE_x,np.asarray(v).reshape(-1,1) ))
        np.savetxt(STAMP+'_TPE_'+str(randomSeed)+'_X.csv', np.asarray(TPE_x), delimiter=",")
        
        pkl_dict = {'TPE_trials':TPE_trials,'TPE_x':np.asarray(TPE_x),'TPE_y':np.asarray(np.asarray(TPE_history).T)}
        if MAXIMIZE_OBJECTIVE==False:
            print('\n TPE minimum',np.min(TPE_history))
            np.savetxt(STAMP+'_TPE_'+str(randomSeed)+'_Y.csv', np.asarray(np.asarray(TPE_history).T), delimiter=",")
        else:
            print('\n TPE maximum',np.max(np.asarray(np.asarray(TPE_history).T)*(-1.)))
            np.savetxt(STAMP+'_TPE_'+str(randomSeed)+'_Y.csv', np.asarray(np.asarray(TPE_history).T)*(-1.), delimiter=",")
        OBJ_NAME = STAMP+'_TPE_'+str(randomSeed)+'.pkl'
        with open(OBJ_NAME, "wb+") as file_model: 
              pkl.dump(pkl_dict, file_model)
    ########################################


    ########################################
    # SMAC (DEFAULT: Min)
    # See its usage for branin https://automl.github.io/SMAC3/master/pages/examples/commandline/branin.html https://automl.github.io/SMAC3/master/pages/details/facades.html
    ########################################
    #
    if 'SMAC' in methodList:
    
        np.random.seed(randomSeed)
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
        # Import ConfigSpace and different types of parameters
        from smac.configspace import ConfigurationSpace
        from smac.facade.smac_hpo_facade import SMAC4HPO, SMAC4AC
        # Import SMAC-utilities
        from smac.scenario.scenario import Scenario

        def SMAC_objective(x):
            h_counter = 0
            x_counter = 0
            continuous_part = []
            categorical_part = []
            for k in x.keys():
                if k[0]=='x':
                    continuous_part.append(x[k])
                    x_counter = x_counter +1
                else:
                    categorical_part.append(x[k])
                    h_counter = h_counter +1
            res = objective(continuous_part,categorical_part)
            print(continuous_part,categorical_part,'f=',res)
            if MAXIMIZE_OBJECTIVE==False:
                return res
            else:
                return -1.*res

        # Scenario object
        print('=====SMAC result=====')
        print(SMAC_space)
        SMAC_scenario = Scenario({
                             "run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": evaluationBudget,  # max. number of function evaluations
                             "cs": SMAC_space,  # configuration space
                             "deterministic": "true",  
                             'abort_on_first_run_crash':"false",
                             "output_dir":WORKING_DIR                       
                             })
        SMAC_scenario.output_dir_for_this_run=WORKING_DIR
        #intensifier_kwargs = {
        #"maxR": 1,  # Each configuration will be evaluated maximal 3 times with various seeds
        #"minR": 0,  # Each configuration will be repeated at least 1 time with different seeds
        #}
        # Optimize, using a SMAC-object
        smac = SMAC4HPO(scenario=SMAC_scenario,
                       rng=RNG,
                       tae_runner=SMAC_objective)
        #smac = SMAC4AC(scenario=SMAC_scenario,
        #               rng=RNG,
        #               tae_runner=SMAC_objective)
        SMAC_best = smac.optimize()

        from smac.runhistory.runhistory import RunKey
        #SMAC_best.trajectory
        #print(SMAC_best.keys)

        SMAC_history_x = []
        SMAC_history = []
        conf_list = smac.solver.runhistory.get_all_configs()
        rec_list = list(smac.solver.runhistory.data.items())

        for j in range(len(rec_list)):
            SMAC_x = conf_list[j]
            rec = rec_list[j][1]
            #print(j,rec)
            SMAC_f = rec.cost
            #print(SMAC_x,SMAC_f)
            SMAC_x_row = []
            for i, (k, v) in enumerate(SMAC_x.items()):
                #print(i, k, len(v))
                #print(np.asarray(v).shape)
                #print(v)
                SMAC_x_row.append(v)
            #print('myr',SMAC_x_row)
            SMAC_history_x.append(SMAC_x_row)
            SMAC_history.append(SMAC_f)
        print(SMAC_history)
        #print('>>>>>total evaluations: ',len(SMAC_history))

        np.savetxt(STAMP+'_SMAC_'+str(randomSeed)+'_X.csv', np.asarray(SMAC_history_x), delimiter=",")
        if MAXIMIZE_OBJECTIVE==False:
            print('MIN--->',np.min(np.asarray(np.asarray(SMAC_history).T)))
            np.savetxt(STAMP+'_SMAC_'+str(randomSeed)+'_Y.csv', np.asarray(np.asarray(SMAC_history).T), delimiter=",")
        else:
            print('MAX--->',np.max(np.asarray(np.asarray(SMAC_history).T*(-1.))))
            np.savetxt(STAMP+'_SMAC_'+str(randomSeed)+'_Y.csv', np.asarray(np.asarray(SMAC_history).T)*(-1.), delimiter=",")
    ########################################


    ########################################
    # EXP3BO (DEFAULT: min)
    ########################################
    #
    if 'EXP3BO' in methodList:
    
        np.random.seed(randomSeed)
        EXP3BO_path=DESTINATION+'AlgorithmicAssurance_NIPS2018/'
        sys.path.insert(0, EXP3BO_path)

        from MAB.GP_EXP3 import GP_EXP3
        from MAB.RoundRobin_MAB import RoundRobin_MAB
        from MAB.RandomArm_MAB import RandomArm_MAB
        from MAB.Oracle_BO import Oracle_BO
        import testFunctions.syntheticFunctions

        import itertools
        EXP3BO_encode_list = list(itertools.product(*categorical_list))
        print('EXP3BO>>>all possible categorical combinations',EXP3BO_encode_list)
        
        def EXP3BO_objective(ht,x):
            #print(ht,x.shape,'<<<')
            # ht is a categorical index
            # X is a continuous variable
            y_list = []
            for k in range(x.shape[0]):
                continuous_part = x[k,:]
                categorical_part = EXP3BO_encode_list[ht] #The ht passes an encoding from 0 to categoriesN,
                #Converting it into a combination of categorical variable values in EXP3BO_encode_list
                y_list.append( 1*objective(continuous_part,categorical_part) )
            y_list = np.asarray(y_list)
            if MAXIMIZE_OBJECTIVE == False:
                return y_list.reshape(-1,1)
            else:
                return -1.*y_list.reshape(-1,1)

        print('EXP3BO>>>total number of categories',categoriesN)
        #%% Run EXP3 Algorithm
        myexp3 = GP_EXP3(objfn=EXP3BO_objective, initN=initialsN, bounds=EXP3BO_space, 
                         acq_type='LCB', C=categoriesN, rand_seed=randomSeed)
        myexp3.runoptimBatchList(trialsN, evaluationBudget)#-initialsN*categoriesN)

        def yield_from_EXP3BO(myexp3=myexp3,EXP3BO_encode_list=EXP3BO_encode_list):
            C_counter = [0]*myexp3.C
            hist_x = []
            hist_y = []
            for i in range(len(myexp3.X)):
                cat_ind = myexp3.X[i][0]  
                my_ht = list(EXP3BO_encode_list[cat_ind])
                my_xt = list(myexp3.data[cat_ind][C_counter[cat_ind],:])
                my_y  = myexp3.result[cat_ind][C_counter[cat_ind],:]
                #print(my_ht+my_xt,'->',my_y)
                hist_x.append(my_ht+my_xt)
                hist_y.append(my_y)
                C_counter[cat_ind] = C_counter[cat_ind]+1
            if MAXIMIZE_OBJECTIVE == False:
                return np.asarray(hist_x),np.asarray(hist_y)
            else:
                return np.asarray(hist_x),np.asarray(hist_y)*(-1.)
                
        print('=====EXP3BO result=====')
        EXP3BO_hist_x, EXP3BO_hist_y = yield_from_EXP3BO(myexp3,EXP3BO_encode_list)
        np.savetxt(STAMP+'_EXP3BO_'+str(randomSeed)+'_X.csv', EXP3BO_hist_x, delimiter=",")
        np.savetxt(STAMP+'_EXP3BO_'+str(randomSeed)+'_Y.csv', EXP3BO_hist_y, delimiter=",")

        roundrobinMAB = RoundRobin_MAB(objfn=EXP3BO_objective, initN=initialsN, bounds=EXP3BO_space,                acq_type='LCB', C=categoriesN, rand_seed=randomSeed)
        roundrobinMAB.runoptimBatchList(trialsN, evaluationBudget)
        #%% Baseline Round-Robin BO

        roundrobinMAB_hist_x, roundrobinMAB_hist_y = yield_from_EXP3BO(roundrobinMAB,EXP3BO_encode_list)
        np.savetxt(STAMP+'_roundrobinMAB_'+str(randomSeed)+'_X.csv', roundrobinMAB_hist_x, delimiter=",")
        np.savetxt(STAMP+'_roundrobinMAB_'+str(randomSeed)+'_Y.csv', roundrobinMAB_hist_y, delimiter=",")

        randomMAB = RandomArm_MAB(objfn=EXP3BO_objective, initN=initialsN, bounds=EXP3BO_space,            acq_type='LCB', C=categoriesN, rand_seed=randomSeed)
        randomMAB.runoptimBatchList(trialsN, evaluationBudget)
        #%% Baseline Random arm BO

        sys.path.remove(EXP3BO_path)

        randomMAB_hist_x, randomMAB_hist_y = yield_from_EXP3BO(randomMAB,EXP3BO_encode_list)
        np.savetxt(STAMP+'_randomMAB_'+str(randomSeed)+'_X.csv', randomMAB_hist_x, delimiter=",")
        np.savetxt(STAMP+'_randomMAB_'+str(randomSeed)+'_Y.csv', randomMAB_hist_y, delimiter=",")
        
        pkl_dict = {'EXP3BO':myexp3,'roundrobinMAB':roundrobinMAB,'randomMAB':randomMAB}
        OBJ_NAME = STAMP+'_EXP3BO_'+str(randomSeed)+'.pkl'
        with open(OBJ_NAME, "wb+") as file_model: 
              pkl.dump(pkl_dict, file_model)
    ########################################


    ########################################
    # skopt optimizer (DEFAULT: min)
    ########################################
    #
    if any(item.startswith('skopt') for item in methodList):
    
        np.random.seed(randomSeed)
        from skopt import gp_minimize, forest_minimize, dummy_minimize
        from skopt.space.space import Categorical, Integer, Real

        def skopt_func(x):
            categorical_part = x[0:len(categorical_list)]
            continuous_part = x[len(categorical_list):]
            #print(categorical_part,continuous_part)
            if MAXIMIZE_OBJECTIVE == False:
                return objective(continuous_part,categorical_part)
            else:
                return -1.*objective(continuous_part,categorical_part)
                
        print('=====skopt result=====')       
        #skopt_Dummy
        res_dummy = dummy_minimize(skopt_func,                  # the function to minimize
                          skopt_space,      # the bounds on each dimension of x
                          n_calls=evaluationBudget,         # the number of evaluations of f
                          #n_initial_points=initialsN,
                          #n_restarts_optimizer=1,
                          random_state=randomSeed)   # the random seed
        np.savetxt(STAMP+'_skoptDummy_'+str(randomSeed)+'_X.csv', np.asarray(res_dummy.x_iters), delimiter=",")
        if MAXIMIZE_OBJECTIVE == False:
            np.savetxt(STAMP+'_skoptDummy_'+str(randomSeed)+'_Y.csv', np.asarray(res_dummy.func_vals), delimiter=",")
        else:
            np.savetxt(STAMP+'_skoptDummy_'+str(randomSeed)+'_Y.csv', np.asarray(res_dummy.func_vals)*(-1.), delimiter=",")
            
        #skopt_GP
        res_gp = gp_minimize(skopt_func,                  # the function to minimize
                          skopt_space,      # the bounds on each dimension of x
                          acq_func="EI",      # the acquisition function
                          n_calls=evaluationBudget,#-initialsN,         # the number of evaluations of f
                          n_initial_points=initialsN,  # the number of random initialization points
                          n_restarts_optimizer=1,
                          n_points=1, #You are not allowed to sample 10000 points for f_black but for the acquisition
                          noise=1e-3,       # the noise level (optional)
                          random_state=randomSeed)   # the random seed
        np.savetxt(STAMP+'_skoptGP_'+str(randomSeed)+'_X.csv', np.asarray(res_gp.x_iters), delimiter=",")
        if MAXIMIZE_OBJECTIVE == False:
            np.savetxt(STAMP+'_skoptGP_'+str(randomSeed)+'_Y.csv', np.asarray(res_gp.func_vals), delimiter=",")
        else:
            np.savetxt(STAMP+'_skoptGP_'+str(randomSeed)+'_Y.csv', np.asarray(res_gp.func_vals)*(-1.), delimiter=",")
            
        #skopt_forest
        res_forest = forest_minimize(skopt_func,                  # the function to minimize
                          skopt_space,      # the bounds on each dimension of x
                          base_estimator="RF",
                          acq_func="EI",      # the acquisition function
                          n_calls=evaluationBudget,#-initialsN,         # the number of evaluations of f
                          n_initial_points=initialsN,  # the number of random initialization points
                          #n_restarts_optimizer=1,
                          n_points=1, #You are not allowed to sample 10000 points but only 1 point for the categorical space. 
                          random_state=randomSeed)   # the random seed
        np.savetxt(STAMP+'_skoptForest_'+str(randomSeed)+'_X.csv', np.asarray(res_forest.x_iters), delimiter=",")
        np.savetxt(STAMP+'_skoptForest_'+str(randomSeed)+'_Y.csv', res_forest.func_vals, delimiter=",")
        if MAXIMIZE_OBJECTIVE == False:
            np.savetxt(STAMP+'_skoptForest_'+str(randomSeed)+'_Y.csv', np.asarray(res_forest.func_vals), delimiter=",")
        else:
            np.savetxt(STAMP+'_skoptForest_'+str(randomSeed)+'_Y.csv', np.asarray(res_forest.func_vals)*(-1.), delimiter=",")
        pkl_dict = {'skoptDummy':res_dummy,'skoptGP':res_gp,'skoptForest':res_forest}
        OBJ_NAME = STAMP+'_skopt_'+str(randomSeed)+'.pkl'
        with open(OBJ_NAME, "wb+") as file_model: 
              pkl.dump(pkl_dict, file_model)
    ########################################


    #We are done for this loop. Make a sound.
    #print('\a') 
    beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
    beep(randomSeed)

