

import sequence as sq
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from functions_mehrsa import generate_one_hot_list

#importing libraries
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
#%%Reaading all 4 data files
path = r"E:\Research\ML\protein\Affibody" # defining the path where csv files exist
#Be careful to name the sequence columns to "Sequence"

B1 = sq.sequence('B1')
B1.read_seq("Affibody_B1.csv",path)
F1 = sq.sequence("F1")
F1.read_seq("Affibody_F1.csv",path)
B2 = sq.sequence('B2')
B2.read_seq('Affibody_B2_complement.csv',path)
F2 = sq.sequence('F2')
F2.read_seq('Affibody_F2_complement.csv',path)

#%%finding unique sequences 

B1.find_unique()
F1.find_unique()
B2.find_unique()
F2.find_unique()

#%% Filtering the data (if needed)
B1.remove_freq_below_limit(10)
F1.remove_freq_below_limit(5)
B2.remove_freq_below_limit(10)
F2.remove_freq_below_limit(5)

#%%
df = pd.concat([B1.df_filtered,F1.df_filtered,B2.df_filtered,F2.df_filtered])

#%%
df_unique = df.groupby("Sequence").agg({'count':'sum'}).reset_index()
df_affibody = sq.sequence('df_affibody')
df_affibody.df_unique  = df_unique

#%% Defining a fitness score
df_affibody.get_score3()
df_affibody.df_unique.Score3 = df_affibody.df_unique.Score3/max(df_affibody.df_unique.Score3)

#%%
df_affibody.generate_onehot_list()
df_affibody.df_unique.to_csv('Affibody_data.csv')
#%%

#defining the data which we are buiding our model on it
def data():
   all_data= pd.read_csv('Affibody_data.csv', index_col=False) 
   train, validate, test = np.split(all_data, [int(.6*len(all_data)), int(.8*len(all_data))])
   X_train= pd.DataFrame(generate_one_hot_list(train.Sequence))
   Y_train = train.Score3
   X_val = pd.DataFrame(generate_one_hot_list(validate.Sequence))
   Y_val = validate.Score3
   X_test=  pd.DataFrame(generate_one_hot_list(test.Sequence))
   Y_test = test.Score3
   return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
#defining the model which we want to optimize the hyperparameters in it
#There are two different options either choosing the candidates that are probably promising by choice option
#Or uniform which searches continuesly between the mentioned span
def model(X_train, Y_train, X_val, Y_val):
    
    model = Sequential()
    model.add(Dense({{choice([3654,1827])}}, input_shape=(3654,)))
    model.add(Activation('relu'))  # Activation function can be another hyperparameter to be optimized. However, in here, we have chosen relu 
    
    
    model.add(Dense({{choice([3654,1827,1000])}}))
    model.add(Activation(('relu')))
    
    
    model.add(Dense({{choice([1827,900, 500])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 0.2)}}))
    model.add(Dense({{choice([900, 500, 250,125])}}))
    model.add(Activation(('relu')))
    model.add(Dropout({{uniform(0, 0.2)}}))
    
    model.add(Dense({{choice([ 500, 250,125,50])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 0.2)}}))  
    model.add(Dense({{choice([250,125,50])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 0.2)}})) 
    
    model.add(Dense(1))
    model.add(Activation('relu')) 
    
   
    Adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-4, 10**-5])}})
    model.compile( Adam, loss='mean_squared_error', metrics=['mape']) # metrics and loss could be hyperparameters as well
    model.fit(X_train, Y_train,
              batch_size={{choice([100,150,200,300,400])}},
              epochs={{choice([5,15,20,25,30,40,50])}},
              verbose=2,
              validation_data=(X_val, Y_val))
    score, error = model.evaluate(X_val, Y_val, verbose=0)
    accuracy = cross_val_score(model, X_Train, Y_Train, cv = 4).mean()
    print('Test error:', error)
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}



best_run, best_model = optim.minimize(model=model,# be careful to match the exact name that you are specifying for the model and data part
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=100, # determining the maximum number of runs
                                      trials=Trials())
hyperparams = space_eval(model, best)
#%%
print(best_run)

