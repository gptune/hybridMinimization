#Hengrui Luo
#hrluo@lbl.gov
#2022-01-04
import numpy as np #numpy 1.21.2
import GPy #GPy 1.10.0
#Allow higher jittering (i.e., perturbation on diagonal) when computing
previous_jitchol = GPy.util.linalg.jitchol
GPy.util.linalg.jitchol = lambda A: previous_jitchol(A, maxtries=100)
import scipy #scipy 1.7.1
from scipy.stats import qmc
from termcolor import colored
import copy
import importlib,sys,datetime
import gc
#Custom modules below.
from hybridMinimization.treeClass import *
from hybridMinimization.kernelClass import *
#By default, we would not use the parallel mechanism to speed up the model selection step
#----------Acquisition functions----------#
def maximum_of_surrogate(X, X_sample, Y_sample, gpr, xi=1e-16):
    #if X in X_sample: return np.inf
    overall_mu = np.zeros((X.shape[0],1))
    overall_sigma = np.zeros((X.shape[0],1))
    for gp in gpr:
        my_gp = gp.predict_noiseless(X)
        mu = my_gp[0]
        #print(gp,'^'*100,mu)
        sigma = my_gp[1]
        overall_mu = overall_mu + mu
        overall_sigma = overall_sigma +sigma
    mu = overall_mu/len(gpr)
    return mu[0]
    
def expected_improvement(X, X_sample, Y_sample, gpr, xi=1e-16):
    #if X in X_sample: return np.inf
    overall_mu = np.zeros((X.shape[0],1))
    overall_sigma = np.zeros((X.shape[0],1))
    for gp in gpr:
        my_gp = gp.predict_noiseless(X)
        mu = my_gp[0]
        #print(gp,'^'*100,mu)
        sigma = my_gp[1]
        overall_mu = overall_mu + mu
        overall_sigma = overall_sigma +sigma
    mu = overall_mu/len(gpr)
    sigma = overall_sigma/len(gpr)
    sigma = np.sqrt(np.absolute(sigma)).reshape(-1, 1)
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    mu_sample_opt = np.max(Y_sample)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * scipy.stats.norm.cdf(Z) + sigma * scipy.stats.norm.pdf(Z)
        ei0_idx = np.where(sigma==0.0)
        if len(ei0_idx[0])>0:
            ei[ei0_idx[0]] = 0.0
        #ei[sigma == 0.0] = 0.0
    #print('^'*50,X,'->EI=',np.exp(ei[0]))
    return ei[0]
    
def upper_confidence_bound(X, X_sample, Y_sample, gpr, xi=1e-16):
    overall_mu = np.zeros((X.shape[0],1))
    overall_sigma = np.zeros((X.shape[0],1))
    for gp in gpr:
        my_gp = gp.predict_noiseless(X)
        mu = my_gp[0]
        #print(gp,'^'*100,mu)
        sigma = my_gp[1]
        overall_mu = overall_mu + mu
        overall_sigma = overall_sigma +sigma
    mu = overall_mu/len(gpr)
    sigma = overall_sigma/len(gpr)
    sigma = np.sqrt(np.absolute(sigma)).reshape(-1, 1)
    mu_sample_opt = np.max(Y_sample)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ucb = mu + sigma * xi
        ucb[sigma == 0.0] = 0.0
    return ucb[0]
#------------------------------#

def hybridMinimization(fn,\
                       #fn: the black-box function to be evaluated.
                       selection_criterion = 'custom',fix_model = -1,\
                       #by default choose fix_model=-1 would pick the best model under the selection criterion. 3 for MLP-asin+Matern52 kernel
                       #Here we implement 'loglik', 'AIC', 'BIC', 'HQC'(Hannan–Quinn information criterion) as model selection criteria among CoCaBO and usual GP kernels.
                       #####Categorical variable related parameters#####
                       categorical_list=[[0,1,2],[0,1,2,3,4]],\
                       #a list of categorical choices for each dimension.
                       #    categorical_list: This should be a list of length N_{categorical variables}, each list entry would be a list, consisting of encodings for that specific category.
                       categorical_trained_model=None,\
                       #This has to be a trained model: treeClass. It is unique to hybridM.
                       policy_list=None,update_list=None,exploration_probability=0.10,\
                       #This policy_list is passed to the tree class, setting up search policy for each layer of the tree.
                       #This update_list is passed to the tree class, setting up update strategy for each layer of the tree.
                       #    policy_list: This should be a list of length N_{categorical variables}, only 'UCTS','UCTS_var','EXP3','Multinomial' are supported.
                       #    update_list: This should be a list of length N_{categorical variables}, only 'UCTS','UCTS_var','EXP3','Multinomial' are supported.
                       #    exploration_probability: The probability of random search.
                       #####Continuous variable related parameters#####
                       continuous_list=[[-10,10],[-10,10]],\
                       #a list of bounds for each dimension of continuous vars.
                       #    continuous_list: This should be a list of length N_{continuous variables}, each list entry would be a list, consisting of encodings for bounds of continuous variables.
                       #*****TO-DO: support constrained continuous domain.
                       continuous_trained_model=None,\
                       #This has to be a trained model: GPClass. It is unique to hybridM.
                       #####Hybrid model parameters#####
                       observed_X=None,observed_Y=None,\
                       #These two parameters provide pilot samples to the hybrid model. This would NOT over-ride the categorical_trained_model/continuous_trained_model
                       #    observed_X: this should be n by d matrix, 
                       #    the first N_{categorical variables} columns are categorical variables/encodings; the rest N_{continuous variables} columns are continuous variables.
                       #    observed_Y: this should be n by 1 matrix of response values.
                       n_find_tree=5,n_find_leaf=5,\
                       #n_find_tree: number of searches allowed when trasversing the tree, i.e., how many categorical combinations we want. i.e., budget for categorical variables.
                       #n_find_leaf: number of sequential samples allowed when we reach leaf, i.e., how many continuous variables we're allowed to observe. i.e., batch size for continuous variables.
                       node_optimize='GP-EI',\
                       GP_normalizer = True,\
                       random_seed=123,minimize=True,N_initialize_leaf=0):
                       #node_optimize: the surrogate model at the lead node, currently only 'GP' is supported.
                       #random_seed: set up a random seed for reproducibility. 
                       #minimize: whether we should minimize or maximize the fn.
                       #N_initialize_leaf: whether we get `N_initialize_leaf` samples for each leaf, corresponding to all possible combinations of categorical variables (EXP3BO practice). But this is expensive when there are a lot of combinations.
                       
    #Set random seed
    np.random.seed(random_seed)

    #Minimize or Maximize fn?
    if minimize==False: 
        #raise NotImplementedError('Only minimization is supported at the moment.')
        #return None
        def f_obs(X):
            #noise_var =  np.random.normal(loc=0,scale=np.sqrt(noise_var))
            return +1.*fn(X)
    else:
        def f_obs(X):
            #noise_var =  np.random.normal(loc=0,scale=np.sqrt(noise_var))
            #The GP+MCTS is designed for the reward maximization scheme, therefore in the whole scheme we want to maximize the f_obs.
            return -1.*fn(X)
       
    #Get the dimensions for continuous and categorical part
    X_continuous_dimension = len(continuous_list)
    continuous_bounds = np.array(continuous_list).astype(float)
    X_categorical_dimension = len(categorical_list)
    
    #Proposal functions for GP inner loop.
    def propose_location(acquisition, H_fixed, X_sample, Y_sample, gpr, bounds, n_restarts=1000,seed=42):#2**X_continuous_dimension,seed=42):
        min_val = np.inf
        min_x = np.zeros((1, X_continuous_dimension))
        #np.random.seed(random_seed)
        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -acquisition(np.hstack((H_fixed.reshape(-1,X_categorical_dimension),X.reshape(-1,X_continuous_dimension))), X_sample,\
                    Y_sample, gpr)

        # Sample, then find the best optimum by starting from n_restart different random points.
        ssample_size = n_restarts#1000
        ssample_type = 'uniform'
        if ssample_type == 'LHS':
            sampler = qmc.LatinHypercube(d=X_continuous_dimension, centered=False, seed=seed)
            samplex0 = sampler.random(n=ssample_size)
            l_bounds = bounds[:, 0]
            u_bounds = bounds[:, 1]
            x0_candidate = qmc.scale(samplex0, l_bounds, u_bounds)
        if ssample_type == 'uniform':
            x0_candidate = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(ssample_size, X_continuous_dimension))
        #x0_candidate = np.vstack((x0_candidate,X_sample))#This may be useful when you believe the function is very flat.
        obj_sample = []
        for j in range(x0_candidate.shape[0]):
            obj_sample.append(min_obj(x0_candidate[j,:]))
        x0_best_sample = x0_candidate[np.argmin(obj_sample),:]
        for x0 in [x0_best_sample]:
            res = scipy.optimize.minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B',options={'disp': False,'gtol': 1e-8})        
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x           
        return min_x.reshape(-1, 1), -1*min_val
    
    #Preparation for the categorical part.
    if categorical_trained_model==None:
        categorical_model = setupTree(categorical_list=categorical_list,\
                          policy_list=policy_list,update_list=update_list,exploration_probability=exploration_probability,\
                          print_tree=False)
    else:
        categorical_model = categorical_trained_model
    
    #Preparation for the continuous part.
    if continuous_trained_model==None:
        #For continuous part model since we are only extracting the (X,Y) samples, covariance kernel and the normalizer boolean values, we do not attempt to do 
        H_sample = np.zeros((0,X_categorical_dimension)) #Store the observed values of H locations. 
        X_sample = np.zeros((0,X_continuous_dimension)) #Store the observed values of X locations. 
        Y_sample = np.zeros((0,1)) #Store the eventual output response for the function evaluated at the continuous and categorical part.
    else:
        X_sample = continuous_trained_model.X
        Y_sample = continuous_trained_model.Y        
        GP_normalizer = continuous_trained_model.normalizer
        trained_base_kernel = continuous_trained_model.kern

    #Prepare the results that are to be returned.
    hybrid_history = []
    hybrid_history_x = np.zeros((0,X_categorical_dimension))
    hybrid_history_y = []
    
    #If initialize_leaf = True, we load it to be obseved_X and observed_Y.
    if N_initialize_leaf > 0:
        import itertools
        catL = categorical_list
        conL = continuous_list
        allL = list(itertools.product(*catL))
        allL = np.asarray(allL)
        pickL = np.zeros(shape=(0,len(catL)+len(conL) ))
        for i in range(allL.shape[0]):
            picker = np.random.uniform(low = np.asarray(conL)[:,0],\
                              high = np.asarray(conL)[:,1],size=(1,len(conL)))
            #print(picker,allL[i,:])
            picker_x = np.hstack((np.asarray(allL[i,:]).reshape(1,-1),picker))
            picker_x = picker_x[0]
            print(picker_x)
            pickL = np.vstack((pickL,picker_x))
        #print(pickL)
        pickL_y = np.zeros(shape=(pickL.shape[0],1))
        for j in range(pickL.shape[0]):
            #print(j)
            pickL_y[j,0] = f_obs(pickL[j,:])
        observed_X = pickL
        observed_Y = pickL_y
        
    #If observed samples are provided, we load it into the hybrid model. package: dill 0.3.4
    if (observed_X is not None) & (observed_Y is not None):
        if not observed_X.shape[0]==observed_Y.shape[0]:
            raise AttributeError('ERROR: The observed samples, observed_X and observed_Y should have matched number of rows.')
        for a in range(observed_X.shape[0]):
            current_node = categorical_model
            for b in range(X_categorical_dimension):
                for c in current_node.children:
                    #print(c.word,observed_X[a,0:(b+1)])
                    #print(c.word==observed_X[a,0:(b+1)])
                    if (c.word==observed_X[a,0:(b+1)]).all():
                        current_node = c
                        break
            if not (c.word==observed_X[a,0:(X_categorical_dimension)]).all():
                #print(observed_X[a,0:(X_categorical_dimension)])
                raise ValueError('There is an observation that cannot be assigned to any leaf given the tree: ',observed_X[a,:])
            print(observed_X[a,:],',',observed_Y[a,:],'->',current_node.word,current_node)
            H_this = observed_X[a,0:X_categorical_dimension]
            X_this = observed_X[a,X_categorical_dimension:]
            Y_this = observed_Y[a,:]
            if minimize:
                Y_this = -1*Y_this
            if current_node.isLeaf():
                #print(current_node.word,' is a Leaf.')
                current_node.appendReward(Y_this[0])
                current_node.backPropagate()
            
            hybrid_history_x = np.vstack((hybrid_history_x,observed_X[a,0:(X_categorical_dimension)]))
            # Add observed samples
            H_sample = np.vstack((H_sample, H_this.reshape(-1,X_categorical_dimension)))
            X_sample = np.vstack((X_sample, X_this.reshape(-1,X_continuous_dimension)))
            Y_sample = np.vstack((Y_sample, Y_this.reshape(-1,1)))
    #categorical_model.printTree()
    #print(H_sample,X_sample,Y_sample)
    #return 
    
    # Get a time performance
    now = datetime.datetime.now()
    print('Start Time:',now.strftime("%Y-%m-%d %H:%M:%S"))
    
    #Put a placeholder model_history, recording which models are chosen each time. 
    model_history = []#None
    # Main loop for tree search
    i = -1
    while (i<=n_find_tree) and (Y_sample.shape[0]<n_find_tree*n_find_leaf):
        i = i + 1
        #print(colored("hybridMinimization>>>",'red'))
        print('\n\n\n')
        print('hybridMinimization>>> >>>',Y_sample.shape[0],' out of ',n_find_tree*n_find_leaf,'<<<')
        if minimize:
            print(colored('Current best(minimum): ','white','on_red'),np.amin(-1.*Y_sample))
        else:
            print(colored('Current best(maximum): ','white','on_red'),np.amax(Y_sample))
        #----------Tree search----------#
        #Use the tree structure to find the optimal next categorical variables. 
        best_leaf = categorical_model.findBestLeaf()
        #categorical_model.printTree()
        categorical_vals = best_leaf.word
        H_next = np.asarray(categorical_vals).reshape(-1,X_categorical_dimension)
        #------------------------------#
        
        #----------Leaf search----------#
        if node_optimize=='GP' or node_optimize == 'GP-EI':
            myCHOICE = []
            myREWARD = []
            #----------Setting up the leaf----------#    
            #print('myKER',myKER)
            #------------------------------#
            leaf_str = [str(x) for x in best_leaf.word]
            leaf_str = '->'.join(leaf_str)
            #from scipy.stats import norm
            #from scipy.optimize import minimize
            k = 0
            while (k < n_find_leaf) and (Y_sample.shape[0]<n_find_tree*n_find_leaf):#range(n_find_leaf):
                #Append 1 observations to this leaf's corresponding history
                continuous_model_list = []
                continuous_model_acq = []
                continuous_model_ucb = []
                continuous_model_max = []
                continuous_model_loglik = []
                continuous_model_AIC = []
                continuous_model_BIC = []
                continuous_model_HQC = []
                continuous_model_custom = []
                model_cand = []
                for ii in range(10):              
                    my_enc = None
                    #----------Setting up the kernel----------#
                    if fix_model > 0 and ii!=fix_model:
                        continue
                    else:
                        skip_list = [0,1,2]
                        if ii in skip_list:#Do we want to skip some models?
                            continue
                        else:
                            model_cand.append(ii)
                        my_enc = None
                    
                    #Default continuous part kernel.
                    if ii<=2:
                        K_con_part = GPy.kern.Matern52(input_dim=X_continuous_dimension,\
                        active_dims=list(np.arange(X_categorical_dimension,X_continuous_dimension+X_categorical_dimension) ),ARD=True)
                        #K_con_part.lengthscale.constrain_bounded(1e-4, 3.)
                        #Default categorical part kernel. 
                        K_cat_part = CategoryOverlapKernel(input_dim=X_categorical_dimension,active_dims=list(np.arange(0,X_categorical_dimension)),encoder=my_enc)    
                    
                    if ii==0:
                        #CoCaBO 0.0
                        mix_value = 0.0
                        myKER = MixtureViaSumAndProduct(X_categorical_dimension+X_continuous_dimension,
                                                        K_cat_part, K_con_part, mix=mix_value, fix_inner_variances=True,
                                                        fix_mix=True)
                        ker_str = 'CoCaBO0.0'
                    if ii==1:
                        #CoCaBO 0.5
                        mix_value = 0.5
                        myKER = MixtureViaSumAndProduct(X_categorical_dimension+X_continuous_dimension,
                                                        K_cat_part, K_con_part, mix=mix_value, fix_inner_variances=True,
                                                        fix_mix=True)
                        ker_str = 'CoCaBO0.5'
                    if ii==2:
                        #CoCaBO 1.0
                        mix_value = 1.0
                        myKER = MixtureViaSumAndProduct(X_categorical_dimension+X_continuous_dimension,
                                                        K_cat_part, K_con_part, mix=mix_value, fix_inner_variances=True,
                                                        fix_mix=True)
                        ker_str = 'CoCaBO1.0'
                    if ii==3:
                        #skoptGP with MLP for the categorical variables.
                        myKER = GPy.kern.MLP(input_dim=X_categorical_dimension,\
                                active_dims=list(np.arange(0,X_categorical_dimension)),\
                                ARD=True )+\
                                GPy.kern.Matern52(input_dim=X_continuous_dimension,\
                                active_dims=list(np.arange(X_categorical_dimension,X_continuous_dimension+X_categorical_dimension) ),ARD=True )
                        ker_str = 'MLP-asin+Matern52'
                        
                    if ii==4:
                        #skoptGP with MLP for the categorical variables.
                        myKER = GPy.kern.MLP(input_dim=X_categorical_dimension,\
                                active_dims=list(np.arange(0,X_categorical_dimension)),\
                                ARD=True )*\
                                GPy.kern.Matern52(input_dim=X_continuous_dimension,\
                                active_dims=list(np.arange(X_categorical_dimension,X_continuous_dimension+X_categorical_dimension) ),ARD=True )
                        ker_str = 'MLP-asin*Matern52'
                    if ii==5:
                        #skoptGP with MLP for the categorical variables.
                        myKER = GPy.kern.MLP(input_dim=X_categorical_dimension,\
                                active_dims=list(np.arange(0,X_categorical_dimension)),\
                                ARD=True )+\
                                GPy.kern.Matern52(input_dim=X_continuous_dimension,\
                                active_dims=list(np.arange(X_categorical_dimension,X_continuous_dimension+X_categorical_dimension) ),ARD=True )+\
                                GPy.kern.MLP(input_dim=X_categorical_dimension,\
                                active_dims=list(np.arange(0,X_categorical_dimension)),\
                                ARD=True )*\
                                GPy.kern.Matern52(input_dim=X_continuous_dimension,\
                                active_dims=list(np.arange(X_categorical_dimension,X_continuous_dimension+X_categorical_dimension) ),ARD=True )
                        ker_str = 'MLP-asin+Matern52+MLP-asin*Matern52'
                        
                    if ii==6:
                        #skoptGP with Matern12.
                        myKER = GPy.kern.Exponential(input_dim=X_categorical_dimension+X_continuous_dimension,active_dims=None,ARD=True )#+\
                                #GPy.kern.White(input_dim=X_categorical_dimension+X_continuous_dimension,active_dims=None )
                        ker_str = 'Matern12'
                    if ii==7:
                        #skoptGP with Matern52, its default.
                        myKER = GPy.kern.Matern52(input_dim=X_categorical_dimension+X_continuous_dimension,active_dims=None,ARD=True )#+\
                                #GPy.kern.White(input_dim=X_categorical_dimension+X_continuous_dimension,active_dims=None )
                        ker_str = 'Matern52'
                    if ii==8:
                        #skoptGP with RBF.
                        myKER = GPy.kern.RBF(input_dim=X_categorical_dimension+X_continuous_dimension,active_dims=None,ARD=True )#+\
                                #GPy.kern.White(input_dim=X_categorical_dimension+X_continuous_dimension,active_dims=None )
                        ker_str = 'MaternInf'
                    if ii==9:
                        #skoptGP with MLP or Matern52 for the categorical variables.
                        #myKER1 is the MLP imposed by Matern52 for categorical.
                        #myKER2 is the continuous kernel.
                        myKER1 = GPy.kern.MLP(input_dim=X_categorical_dimension,\
                                active_dims=list(np.arange(0,X_categorical_dimension)),\
                                ARD=True )+\
                                GPy.kern.Matern52(input_dim=X_categorical_dimension,\
                                active_dims=list(np.arange(0,X_categorical_dimension)),\
                                ARD=True )
                        myKER2 = GPy.kern.Matern52(input_dim=X_continuous_dimension,\
                                active_dims=list(np.arange(X_categorical_dimension,X_continuous_dimension+X_categorical_dimension) ),ARD=True )
                        myKER = myKER1 + myKER2 
                        ker_str = 'K1 = MLP-asin; K2 = Matern52(for cat)+Matern52(for con); K1+K2'    

                    print('hybridMinimization>>>Trying model ',ii,' >>> Kernel:',ker_str)
                    
                    #We want to use the idx_i part of the observed samples.
                    idx_i = range(X_sample.shape[0]) 
                    continuous_model = GPy.models.GPRegression(np.hstack((H_sample[idx_i,:],X_sample[idx_i,:])), Y_sample, kernel=myKER, normalizer=GP_normalizer)
                    tmp_model = continuous_model.copy()
                    continuous_model.Gaussian_noise.variance.constrain_bounded(1e-6, 1e-2)
                    try:
                        continuous_model.optimize(optimizer='lbfgs', messages=False, max_iters=1e3)
                        #continuous_model.optimize_restarts(num_restarts = 3)
                    except (RuntimeError, TypeError, ValueError,np.linalg.LinAlgError): 
                        print('Model ',ii,' is numerically unstable in optimization.')
                        continuous_model = tmp_model

                    if selection_criterion=='loglik':
                        continuous_model_loglik.append(continuous_model.log_likelihood())
                        #loglik = LL
                        #the model with the highest loglik is selected.
                        #choose minimum -L <==> choose maximum L. This is almost like the minimal description length (MDL) when we use uniform prior on parameters (c.f. ESL2).
                    if selection_criterion=='AIC':
                        N_s = continuous_model.X.shape[0]
                        myAIC = (2/N_s)*len(continuous_model.parameter_names())-(2/N_s)*continuous_model.log_likelihood()
                        #AIC = 2/n * k -2/n * LL, the model with the lowest AIC is selected.
                        continuous_model_AIC.append(-1.*myAIC)
                    if selection_criterion=='BIC':
                        N_s = continuous_model.X.shape[0]
                        myBIC = len(continuous_model.parameter_names())*np.log(N_s)-2*continuous_model.log_likelihood()
                        #BIC = log(n) * k - 2 * LL, the model with the lowest BIC is selected.
                        continuous_model_BIC.append(-1.*myBIC)
                    if selection_criterion=='HQC':
                        N_s = continuous_model.X.shape[0]
                        myHQC = 2*len(continuous_model.parameter_names())*np.log(np.log(N_s))-2*continuous_model.log_likelihood()
                        #HQC = log(log(n)) * 2k - 2 * LL, the model with the lowest HQC is selected.
                        continuous_model_HQC.append(-1.*myHQC)
                    if selection_criterion=='acq':
                        X_next, best_acq = propose_location(acquisition=expected_improvement,H_fixed=H_next,\
                                                            X_sample=X_sample, Y_sample=Y_sample, gpr=[continuous_model], bounds=continuous_bounds,seed=random_seed)
                        continuous_model_acq.append(best_acq)
                        #the model with the highest acquisition function value is selected.
                    if selection_criterion=='custom':
                        continuous_model_loglik.append(continuous_model.log_likelihood())
                        X_next, best_acq = propose_location(acquisition=expected_improvement,H_fixed=H_next,\
                                                            X_sample=X_sample, Y_sample=Y_sample, gpr=[continuous_model], bounds=continuous_bounds,seed=random_seed)
                        continuous_model_acq.append(best_acq)
                        #X_next, best_ucb = propose_location(acquisition=expected_improvement,H_fixed=H_next,\
                        #                                    X_sample=X_sample, Y_sample=Y_sample, gpr=[continuous_model], bounds=continuous_bounds,seed=random_seed)
                        #continuous_model_ucb.append(best_ucb)
                        #X_next, best_sur = propose_location(acquisition=maximum_of_surrogate,H_fixed=H_next,\
                        #                                    X_sample=X_sample, Y_sample=Y_sample, gpr=[continuous_model], bounds=continuous_bounds,seed=random_seed)
                        #continuous_model_max.append(best_sur)
                        
                    #Compute corresponding model.
                    continuous_model_list.append(copy.deepcopy(continuous_model))
                    #Garbage collection.
                    del continuous_model
                    del tmp_model
                    gc.collect()
                
                if selection_criterion=='loglik':
                    criterion_list = continuous_model_loglik
                if selection_criterion=='AIC':
                    criterion_list = continuous_model_AIC
                if selection_criterion=='BIC':
                    criterion_list = continuous_model_BIC 
                if selection_criterion=='HQC':
                    criterion_list = continuous_model_HQC    
                if selection_criterion=='acq':
                    #c.f. [Gustavo Malkomes,Chip Schaff,Roman Garnett] Bayesian optimization for automated model selection, 2016.
                    criterion_list = continuous_model_acq  
                if selection_criterion=='custom':
                    #standardized so that loglik and acq are on the same [-1,1] scale
                    standard1 = np.asarray(continuous_model_loglik)#/np.max(np.abs(continuous_model_loglik))
                    standard2 = np.asarray(continuous_model_acq)#/np.max(np.abs(continuous_model_acq))
                    #standard3 = np.asarray(continuous_model_max)- np.max(Y_sample)
                    #standard4 = np.asarray(continuous_model_ucb)
                    print('s1',np.round(standard1,3))
                    print('s2',np.round(standard2,3))
                    #print('s3',np.round(standard3,3))
                    #print('s4',np.round(standard4,3))
                    
                    #criterion_list = standard1.argsort().argsort() + 0.5*standard2.argsort().argsort()
                    criterion_list = standard1.argsort().argsort() + 2*i/n_find_tree*standard2.argsort().argsort()
                ############################################################    
                #This section is for debugging and experimental only, do not change if you are a regular user. 
                #Pick the best model accrding to the model selection criteria.
                if fix_model>0:
                    arg_idx = fix_model
                    #print('hybridMinimization>>>FIX MODEL',arg_idx)
                    #continuous_model_list = [continuous_model_list[fix_model]]
                else:
                    if i==0:
                        arg_idx = 0
                        #continuous_model_list = [continuous_model_list[0]]
                    else:
                        arg_idx = [index for index, value in sorted(enumerate(criterion_list), reverse=True, key=lambda x: x[1]) if value > -np.inf][:np.abs(fix_model)]
                        #e.g., fix_model = -2, meaning we aggregate the top-2 surrogate models.
                        #This is an experimental function, do not change if you do not understand your purpose.
                #print(arg_idx,continuous_model_list)
                ############################################################
                if fix_model>0:
                    print(colored('hybridMinimization>>>','white','on_blue'),'I deterministically choose model ',arg_idx,' from model selection criteria list',np.round(criterion_list,3))
                else:
                    if isinstance(arg_idx, list):
                        continuous_model_list = [continuous_model_list[idx] for idx in arg_idx]
                        model_history.append(arg_idx)
                        print(colored('hybridMinimization>>>','white','on_blue'),'Based on ',selection_criterion,', I automatically choose maximum',[model_cand[iii] for iii in arg_idx],' from model selection criteria list\n',np.round(criterion_list,3))
                    else:
                        continuous_model_list = [continuous_model_list[arg_idx]]
                        model_history.append([arg_idx])
                        print(colored('hybridMinimization>>>','white','on_blue'),'Based on ',selection_criterion,', I automatically choose maximum',model_cand[arg_idx],' from model selection criteria list\n',np.round(criterion_list,3))
                            
                if node_optimize == 'GP' or node_optimize == 'GP-EI':
                    X_next, best_acq = propose_location(acquisition=expected_improvement,H_fixed=H_next,\
                            X_sample=X_sample, Y_sample=Y_sample, gpr=continuous_model_list, bounds=continuous_bounds,seed=random_seed)
                else:
                    raise NotImplementedError('Continuous optimizer: ',node_optimize,' has not been implemented.')
                
                #Form the full X variable for the next sampling location, including both categorical and continuous part. 
                #print(categorical_vals)
                X_next_full = categorical_vals.copy()
                for itm in list(X_next):
                    X_next_full.append(itm[0])
                
                # Obtain next noisy sample from the objective function
                Y_next = f_obs(X_next_full)
                
                #If we get improvement, get another shot.
                print('>>>>> >>>>>k=',int(k),'/',n_find_leaf,sep='')
                k = k + 1
                
                #print(X_next,Y_next)
                # Add sample to previous samples
                H_sample = np.vstack((H_sample, H_next.reshape(-1,X_categorical_dimension)))
                X_sample = np.vstack((X_sample, X_next.reshape(-1,X_continuous_dimension)))
                Y_sample = np.vstack((Y_sample, Y_next.reshape(-1,1)))
                
                myCHOICE.append(X_next)
                myREWARD.append(Y_next.ravel())
        else:
            raise NotImplementedError('Continuous optimizer: ',node_optimize,' has not been implemented.')
            return None
        
        best_val = np.mean(myREWARD)
        best_opt = myCHOICE[np.argmax(myREWARD)]
        # For a full configuration of categorical+continuous values
        cat_vals = categorical_vals.copy()
        full_vals = categorical_vals.copy()
        for itm in list(best_opt):
            full_vals.append(itm[0])
        print('hybridMinimization>>> next sample>',np.round(np.asarray(full_vals),3),np.round(Y_next,3))
        #------------------------------#
        
        hybrid_history.append(np.max(myREWARD))
        hybrid_history_x = np.vstack(( hybrid_history_x, np.tile(cat_vals, (n_find_leaf,1)) ))
        hybrid_history_y.append(Y_next.ravel())
        best_leaf.appendReward(best_val)
        best_leaf.backPropagate()
        #print(categorical_vals,np.round(best_val,6),',achieved at x=',best_opt,' at leaf ',best_leaf.getWord())
        #categorical_model.printTree()
   
    now = datetime.datetime.now()
    print('End Time:',now.strftime("%Y-%m-%d %H:%M:%S"))
    
    aug_X_sample = np.hstack( (H_sample,X_sample) )
    if minimize==False:
        return +1*Y_sample,aug_X_sample,categorical_model,continuous_model_list,model_history
    else:
        return -1*Y_sample,aug_X_sample,categorical_model,continuous_model_list,model_history

