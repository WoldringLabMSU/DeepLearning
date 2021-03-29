
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
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
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
#Data should be reshaped to be compatible with CNN 
X_train = pd.DataFrame(X_train).to_numpy()
X_test = pd.DataFrame(X_test).to_numpy()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#%% 1D CNN which contains convolution layer, pooling layer and dense layer

model = Sequential()

model.add(Conv1D(100, 10, activation='relu', input_shape=(1218,1)))
model.add(MaxPooling1D(pool_size=2)) # Averagepooling could be used, as well

model.add(Conv1D(100, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) 

model.add(Conv1D(100, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) 

model.add(Conv1D(50, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) 


model.add(Conv1D(50, kernel_size = 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) 

model.add(Dense(25, activation='relu'))
    
model.add(Dense(1, activation='relu'))
model.compile(Adam(lr = 0.0001, decay = 0.003), loss='mean_absolute_error', metrics=['mape'])
                
model.fit(X_train,Y_train,epochs=20, batch_size =100,verbose = 2, validation_split=0.2)

Prediction = model.predict(X_test, verbose = 2) 