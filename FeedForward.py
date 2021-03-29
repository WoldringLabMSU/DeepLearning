
#Importing the required libraries

import keras
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import Model
from keras.layers.core import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.activations import relu, elu
from keras.layers import  Dropout
from keras.activations import sigmoid
from keras.layers.convolutional import *
from keras.layers.pooling import *
import sys
import data_processing_class as process
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
filter1= 15
filter2 = 10
B1.remove_freq_below_limit(filter1)
F1.remove_freq_below_limit(filter2 )
B2.remove_freq_below_limit(filter1)
F2.remove_freq_below_limit(filter2 )
#%%concatenating the small datasets to have one single united dataset
df = pd.concat([B1.df_filtered,F1.df_filtered,B2.df_filtered,F2.df_filtered])

#%%
df_unique = df.groupby("Sequence").agg({'count':'sum'}).reset_index()
df_affibody = process.sequence('df_affibody')
df_affibody.df_unique  = df_unique
#%%
df_affibody.get_score4()
df_affibody.df_unique.Score4 = df_affibody.df_unique.Score4/max(df_affibody.df_unique.Score4)
#%%
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
"""
The following is the code for implementing the feedforward neural network 
The hyperparameters should be optimized based on the provided data( optimize via OPTUNA or HYPEROPT)
"""
model = Sequential()

model.add(Dense(1218, activation='relu', input_shape=(1218,  ))) #The input shape is based on the protein of interest, here 1218 is (58( AminoAcid seq in affibody) *21(20AAs + -))

model.add(Dense(609, activation='relu'))


model.add(Dense(304, activation='relu'))
model.add(Dropout(0.2))  #  reducing the potential for overfitting
model.add(Dense(152, activation='relu'))

model.add(Dense(76, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(38, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(19, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(Adam(lr = 0.0001), loss='mean_squared_error', metrics=['mape'])
                
model.fit(X_train,Y_train, batch_size =100,epochs=20,verbose = 2, shuffle= True) #epoch and batch size should be optimized

model.summary()
Prediction = model.predict(X_test, verbose = 2)