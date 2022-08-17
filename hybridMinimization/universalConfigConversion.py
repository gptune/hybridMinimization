#Universal Configuration Space Conversion Utils
# Hengrui Luo 2021-10-20
import numpy as np
import hyperopt
from hyperopt import hp
import smac
from smac.configspace import ConfigurationSpace
import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
import sklearn, skopt
from skopt.space import Space
from skopt.space.space import Categorical, Integer, Real

def list_to_config(categorical_list,continuous_list,target_format):
    if target_format=='EXP3BO':
        #list format to EXP3BO format.
        output_config = []
        for j in range(len(continuous_list)):
            tmp_dict = {'name':None,'type':None,'domain':None}
            tmp_dict['name'] = 'x'+str(j)
            tmp_dict['type'] = 'continuous'
            tmp_dict['domain'] = tuple(continuous_list[j])
            print(tmp_dict)
            output_config.append(tmp_dict)    
        #print(output_config)

    elif target_format=='CoCaBO':
        #list format to CoCaBO format.
        output_config = []
        for i in range(len(categorical_list)):
            tmp_dict = {'name':None,'type':None,'domain':None}
            tmp_dict['name'] = 'h'+str(i)
            tmp_dict['type'] = 'categorical'
            tmp_dict['domain'] = tuple(categorical_list[i])
            print(tmp_dict)
            output_config.append(tmp_dict)
        for j in range(len(continuous_list)):
            tmp_dict = {'name':None,'type':None,'domain':None}
            tmp_dict['name'] = 'x'+str(j)
            tmp_dict['type'] = 'continuous'
            tmp_dict['domain'] = tuple(continuous_list[j])
            print(tmp_dict)
            output_config.append(tmp_dict)    
        #print(output_config)

    elif target_format=='TPE':
        #list format to TPE
        from hyperopt import hp
        output_config = dict()
        for i in range(len(categorical_list)):
            output_config['h'+str(i)] = hp.choice('h'+str(i),categorical_list[i])
        for j in range(len(continuous_list)):
            output_config['x'+str(j)] = hp.uniform('x'+str(j),continuous_list[j][0],continuous_list[j][1] ) 
        print(output_config)

    elif target_format=='SMAC' or target_format=='SMAC3':
        #list format to SMAC(SMAC3)
        from smac.configspace import ConfigurationSpace
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
        output_config = ConfigurationSpace()
        for i in range(len(categorical_list)):
            tmp_cat = CategoricalHyperparameter('h'+str(i), choices=categorical_list[i])
            output_config.add_hyperparameters([tmp_cat])
        for j in range(len(continuous_list)):
            tmp_con = UniformFloatHyperparameter('x'+str(j), continuous_list[j][0], continuous_list[j][1],\
                                                 default_value=np.mean(continuous_list[j]))
            output_config.add_hyperparameters([tmp_con])
        #print(output_config)
        
    elif target_format=='sklearn' or target_format=='skopt':
        #list format to skopt/autotune
        tmp_list = []
        for i in range(len(categorical_list)):
            tmp_list.append( Categorical(categorical_list[i], name='h'+str(i)) )
        for j in range(len(continuous_list)):
            tmp_list.append( Real(continuous_list[j][0],continuous_list[j][1], name='x'+str(j)) )
        output_config = Space(tmp_list)
    else:
        raise NotImplementedError('Target format not supported.')
    return output_config
    
def config_to_list(config_space,source_format):
    output_categorical_list = []
    output_continuous_list = []
    
    if source_format=='CoCaBO' or source_format=='EXP3BO':
        #Convert the CoCaBO or EXP3BO configuration space into lists.
        for k in range(len(config_space)):
            if config_space[k]['type']=='categorical':
                output_categorical_list.append( list(config_space[k]['domain']) )
            elif config_space[k]['type']=='continuous':
                output_continuous_list.append( list(config_space[k]['domain']) )
                
    elif source_format=='TPE':
        #Convert the TPE configuration space into lists.
        for k,v in enumerate(config_space):
            param_type = config_space[v].name
            param_obj  = config_space[v].pos_args
            tmp_list = []
            if param_type == 'switch':
                for i in range(1,len(param_obj)):
                    tmp_list.append(param_obj[i]._obj)
                output_categorical_list.append(tmp_list)
            elif param_type == 'float':
                #print('lower bound',param_obj[0].pos_args[1].pos_args[0].obj)
                #print('upper bound',param_obj[0].pos_args[1].pos_args[1].obj)
                lwb = param_obj[0].pos_args[1].pos_args[0].obj
                upb = param_obj[0].pos_args[1].pos_args[1].obj
                output_continuous_list.append([lwb,upb])
                
    elif source_format=='SMAC' or source_format=='SMAC3':     
        #Convert the SMAC(SMAC3) configuration space into lists.
        for k,v in enumerate(config_space._hyperparameters):
            #print(k,v,config_space._hyperparameters[v])
            rec = config_space._hyperparameters[v]
            if isinstance(rec,ConfigSpace.hyperparameters.CategoricalHyperparameter):
                #print(rec.choices)
                output_categorical_list.append(list(rec.choices))
            elif isinstance(rec,ConfigSpace.hyperparameters.UniformFloatHyperparameter):
                #print(rec.lower,rec.upper)
                output_continuous_list.append([rec.lower,rec.upper])
                
    elif source_format=='sklearn' or source_format=='skopt':
        #Convert the skopt/autotune configuration space into lists.
        for k in range(len(config_space.dimensions)):
            rec = config_space.dimensions[k]
            if isinstance(rec,skopt.space.space.Categorical):
                #print('cat',rec.categories)
                output_categorical_list.append(list(rec.categories))
            elif isinstance(rec,skopt.space.space.Integer):
                print('int',rec.bounds,' Integer type not supported.')
            elif isinstance(rec,skopt.space.space.Real):
                #print('uni',rec.bounds)
                output_continuous_list.append(list(rec.bounds))
                
    else:
        raise NotImplementedError('Target format not supported.')
    return output_categorical_list,output_continuous_list

