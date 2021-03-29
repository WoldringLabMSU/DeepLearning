
#Importing the required libraries

import keras
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.optimizers import Adam
import data_processing_class as process
from keras.layers import LSTM
#%%Reaading all 4 data files
path = r"E:\Research\OneDrive - Michigan State University\Research\ML\protein\Affibody" #path where the files are stored  # Affibody id protein o
#This is an example when dealing with different populations ratehr than one unique fasta file. Different NGS reads are defined as an instance of our new class 
B1 = process.sequence('B1') 
B1.read_seq("Affibody_B1_AA.csv",path)
F1 = process.sequence("F1")
F1.read_seq("Affibody_F1_AA.csv",path)
B2 = process.sequence('B2')
B2.read_seq('Affibody_B2_complement_AA.csv',path)
F2 = process.sequence('F2')
F2.read_seq('Affibody_F2_complement_AA.csv',path)


#%%finding unique sequences 

B1.find_unique()
F1.find_unique()
B2.find_unique()
F2.find_unique()

#%% Filtering the data with specific cut-off value
filter1= 10
filter2 = 5
B1.remove_freq_below_limit(filter1)
F1.remove_freq_below_limit(filter2)
B2.remove_freq_below_limit(filter1)
F2.remove_freq_below_limit(filter2)
#%%concatenating the small datasets to have one single united dataset
df = pd.concat([B1.df_filtered,F1.df_filtered,B2.df_filtered,F2.df_filtered])

#%%
df_unique = df.groupby("Sequence").agg({'count':'sum'}).reset_index()
df_affibody = process.sequence('df_affibody')
df_affibody.df_unique  = df_unique
#%%Yeo-Johnson
df_affibody.get_score4()
df_affibody.df_unique.Score4 = df_affibody.df_unique.Score4/max(df_affibody.df_unique.Score4)
#%%Box-Cox
df_affibody.get_score3()
df_affibody.df_unique.Score3 = df_affibody.df_unique.Score3/max(df_affibody.df_unique.Score3)

#%% The one-hot coded sequences will be stored in self.onehot
df_affibody.generate_onehot_list()
#%%Determining the features and their value in order to feed the algorithm 
features = df_affibody.one_hot
labels = df_affibody.df_unique.Score3
#%%
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#%%
# reshape input into [samples, timesteps, features]
look_back = 1
n_features= X_train.shape[1]
X_train = X_train.values.reshape(X_train.shape[0], look_back, n_features)
X_test=  X_test.values.reshape(X_test.shape[0], look_back,n_features)
#%%
model = Sequential()  
model.add(keras.layers.wrappers.Bidirectional(LSTM(200, input_shape=( 100,1218), return_sequences=False))) #reurn sequences will be true if the next layer be directional
model.add(Dropout(0.3))
model.add((Dense(100, activation ='relu')))
model.add(Dropout(0.2))
model.add(Dense(50,  activation ='relu')) 
model.add(Dropout(0.2))
model.add(Dense(20,  activation ='relu'))
model.add(Dense(10,  activation ='relu'))
model.add(Dense(1, activation ='relu'))   
model.compile(Adam(lr = 0.001, decay =0.005), loss='mean_absolute_error', metrics=['mape'])  
                  
model.fit(X_train,Y_train, batch_size =300,epochs=10,verbose = 2, shuffle= True,validation_split=0.1)

Prediction = model.predict(X_test, verbose = 2)
error = model.evaluate(X_test, Y_test)