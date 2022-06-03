########################################
# Benchmark application-defined function AND range
########################################
import tensorflow as tf
import numpy as np
import warnings, time
from sklearn.datasets import load_boston

def get_NN_error(n_model_cat = 3,n_in_dense_layer = 7,learning_rate_p = 0.01,\
                batch_s=8,activation = 'relu',dropout_rate=0.0):
    np.random.seed(42)
    tf.random.set_seed(42)#for reproducibility
    assert n_model_cat in [3,4,5]
    assert n_in_dense_layer >=4 and n_in_dense_layer<=16
    assert learning_rate_p >= 10**(-5) and learning_rate_p<=1
    assert batch_s >= 8 and batch_s <= 64 
    assert activation in ['relu','tanh','sigmoid','linear']
    assert dropout_rate >=0 and dropout_rate < 1
    #load boston dataset
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X, y = load_boston(return_X_y=True)
    #print(X.shape)

    train_X = X
    train_Y = y
    assert train_X.shape[0] == train_Y.shape[0]
    n_train = train_X.shape[0]
    #
    input_layer = tf.keras.layers.Input(shape=(train_X.shape[1]))
    #n_model_cat = 3 #can be 3,4,5, categorical
    #n_in_dense_layer = 7 #can be 4 to 16, integer
    #learning_rate_p = 0.01 #can be 10**-5 to 0.1, continuous
    #batch_s = 8 #can be 8 to 64, integer
    #activation = 'relu' #can be relu/tanh/sigmoid, categorical
    n_BatchSize = batch_s
    if activation == 'relu':
        activation_fun = tf.keras.activations.relu
    if activation == 'tanh':
        activation_fun = tf.keras.activations.tanh
    if activation == 'sigmoid':
        activation_fun = tf.keras.activations.sigmoid
    if activation == 'linear':
        activation_fun = tf.keras.activations.linear
    if n_model_cat == 3:
        #model 1, 3-layer
        fullyconnected_layer = tf.keras.layers.Dense(n_in_dense_layer,activation=activation_fun)(input_layer)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(fullyconnected_layer)
        output_layer = tf.keras.layers.Dense(1)(dropout_layer)
    if n_model_cat == 4:
        #model 2, 4-layer
        fullyconnected_layer1 = tf.keras.layers.Dense(n_in_dense_layer,activation=activation_fun)(input_layer)
        fullyconnected_layer2 = tf.keras.layers.Dense(n_in_dense_layer,activation=activation_fun)(fullyconnected_layer1)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(fullyconnected_layer2)
        output_layer = tf.keras.layers.Dense(1)(dropout_layer)
    if n_model_cat == 5:
        #model 3, 5-layer
        fullyconnected_layer1 = tf.keras.layers.Dense(n_in_dense_layer,activation=activation_fun)(input_layer)
        fullyconnected_layer2 = tf.keras.layers.Dense(n_in_dense_layer,activation=activation_fun)(fullyconnected_layer1)
        fullyconnected_layer3 = tf.keras.layers.Dense(n_in_dense_layer,activation=activation_fun)(fullyconnected_layer2)
        dropout_layer = tf.keras.layers.Dropout(dropout_rate)(fullyconnected_layer3)
        output_layer = tf.keras.layers.Dense(1)(dropout_layer)

    model = tf.keras.models.Model(input_layer,output_layer)
    #optimizers and loss function
    
    startTIME = time.time()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_p),loss=tf.keras.losses.mse)
    history = model.fit(train_X, # Features
                        train_Y, # Target vector
                        epochs=50, # Number of epochs
                        verbose=0, # No output
                        batch_size=n_BatchSize)#, # Number of observations per batch
                        #validation_data=(train_X, train_Y)) # Data for evaluation
    f1 = history.history
    f1_train_X = model.predict(train_X)
    endTIME = time.time()
    print('Elapsed time (fit-predict): ',endTIME - startTIME, ' seconds.')
    Err_T_X = sum((train_Y-f1_train_X)**2)[0]/n_train 
    print('\n n_model_cat(model type/total layer)',n_model_cat,'\n n_in_dense_layer(number of nodes in dense layer)',n_in_dense_layer,\
          '\n learning_rate_p(laerning rate)',learning_rate_p,'\n batch_s(batch size in training)',batch_s,'\n activation(activation function)',activation,\
          '\n dropout_rate(last layer dropout rate, 0= no dropout)',dropout_rate,\
          '\n Training Error (objective)=',Err_T_X)
    #Save some .pkl files for reproducibility.
    return Err_T_X
#get_NN_error(n_model_cat = 3,n_in_dense_layer = 12,learning_rate_p = 0.01,\
#                batch_s=8,activation = 'relu',dropout_rate=0.0)
                
global EXPERIMENT_NAME
EXPERIMENT_NAME = 'NN_boston'
##Benchmark function (We want to maximize this function when MAXIMIZE_OBJECTIVE=True)
def objective(continuous_part,categorical_part):
    categorical_part = np.array(categorical_part).reshape(1,-1)
    continuous_part = np.array(continuous_part).reshape(1,-1)
    z = np.hstack((continuous_part,categorical_part))
    print(continuous_part,categorical_part,z)
    res = 0
    n_model_cat = categorical_part[0,0]
    if categorical_part[0,1] == 0:
        activation = 'relu'
    if categorical_part[0,1] == 1:
        activation = 'tanh'
    if categorical_part[0,1] == 2:
        activation = 'sigmoid'   
    if categorical_part[0,1] == 3:
        activation = 'linear'    
    n_in_dense_layer = np.floor(continuous_part[0,0]).astype(int)
    learning_rate_p = float(continuous_part[0,1])
    batch_s = np.floor(continuous_part[0,2]).astype(int)
    dropout_rate = float(continuous_part[0,3])
    res = get_NN_error(n_model_cat = n_model_cat,n_in_dense_layer = n_in_dense_layer,learning_rate_p = learning_rate_p,\
                       batch_s=batch_s,activation = activation,dropout_rate=dropout_rate)
    print('res =', res)
    return -1.*res #We want to minimize the training error, or equivalently, maximizing its negative.
#   
global categorical_list 
categorical_list = [[3,4,5],[0,1,2,3]]
global continuous_list
continuous_list = [[4,16],[0.00001,1],[8,64],[0,0.5]]
global MAXIMIZE_OBJECTIVE
MAXIMIZE_OBJECTIVE = True
global NUM_OF_REPEATS
NUM_OF_REPEATS = 10
global evaluationBudget
evaluationBudget = 100 #number of evaluations 
global hybridStrategy
hybridStrategy = 'UCTS'
global methodList 
methodList = ['hybridM'] 
