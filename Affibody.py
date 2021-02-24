# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:54:54 2020

@author: me2ma
"""

# import sequence as sq
# import keras
# import pandas as pd
# import tensorflow as tf
# from keras.models import Sequential
# from tensorflow.keras import Model
# from keras.layers.core import Dense
# import numpy as np
# import matplotlib.pyplot as plt  
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from keras.layers import Dropout
# from keras.optimizers import Adam
# from keras.activations import relu, elu
# from keras.layers import  Dropout
# from keras.activations import sigmoid
# from sklearn.model_selection import KFold, StratifiedKFold

# from keras.layers.convolutional import *
# from keras.layers.pooling import *

# from keras.layers import LSTM
from tensorflow import keras
import keras
from keras.models import Sequential
from tensorflow.keras import Model
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.activations import relu, elu
from keras.activations import sigmoid
import pandas as pd
import numpy as np
import sequence as sq
#%%Reaading all 4 data files
path = r"C:\Users\me2ma\OneDrive\Desktop\Research\ML\protein\Affibody"

B1 = sq.sequence('B1')
B1.read_seq("Affibody_B1_AA.csv",path)
F1 = sq.sequence("F1")
F1.read_seq("Affibody_F1_AA.csv",path)
B2 = sq.sequence('B2')
B2.read_seq('Affibody_B2_complement_AA.csv',path)
F2 = sq.sequence('F2')
F2.read_seq('Affibody_F2_complement_AA.csv',path)

#%%finding unique sequences and removing underlines

B1.find_unique()
F1.find_unique()
B2.find_unique()
F2.find_unique()
B1.remove_underline()
F1.remove_underline()
B2.remove_underline()
F2.remove_underline()
#%% Filtering the data (if needed)
B1.remove_freq_below_limit(15)
F1.remove_freq_below_limit(10)
B2.remove_freq_below_limit(15)
F2.remove_freq_below_limit(10)

#%%Normalizing and finding the commons and exclusives
F1.normalize_within_library(B1)
B1.get_common_unique_filtered_seq(F1)
B1.get_exclusive_unique_filtered_seq(F1)
F1.get_common_unique_filtered_seq(B1)
F1.get_exclusive_unique_filtered_seq(B1)

F2.normalize_within_library(B2)
B2.get_common_unique_filtered_seq(F2)
B2.get_exclusive_unique_filtered_seq(F2)
F2.get_common_unique_filtered_seq(B2)
F2.get_exclusive_unique_filtered_seq(B2)
#%% Get the scores and the score represented in each object is the summation of scores for both technniques in one library
B1.get_score1(F1)
F1.get_score1(B1)

B2.get_score1(F2)
F2.get_score1(B2)

#%% Getting the onehot encoded sequences 
B1.generate_onehot_list1()
F1.generate_onehot_list1()
B2.generate_onehot_list1()
F2.generate_onehot_list1()
#%%Normalize scores within dataset


all_data = pd.concat([B1.df_all, B2.df_all])
all_data.FitnessScore = all_data.FitnessScore/ max (all_data.FitnessScore)

out= all_data.Sequence   
out.to_csv('Affibody_all_unirep.csv', index=False)                              
#%%
features = pd.DataFrame(generate_one_hot_list(all_data.Sequence))
labels = all_data.FitnessScore
#%%
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#%%
df = pd.concat([B1.df_filtered,F1.df_filtered,B2.df_filtered,F2.df_filtered])

#%%
df_unique = df.groupby("Sequence").agg({'count':'sum'}).reset_index()
df_affibody = sq.sequence('df_affibody')
df_affibody.df_unique  = df_unique
#%%finding unique sequences and removing underlines
df_affibody.get_score2()
df_affibody.df_unique.Score2 = df_affibody.df_unique.Score2/max(df_affibody.df_unique.Score2)
#%%
df_affibody.get_score3()
df_affibody.df_unique.Score3 = df_affibody.df_unique.Score3/max(df_affibody.df_unique.Score3)
#%%
df_affibody.generate_onehot_list2()
#%%
df_affibody.get_score4()
df_affibody.df_unique.Score4 = df_affibody.df_unique.Score4/max(df_affibody.df_unique.Score4)
#%%
features = df_affibody.one_hot
labels = df_affibody.df_unique.Score4
#%%
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#%%
train, validate, test = np.split(all_data, [int(.6*len(all_data)), int(.8*len(all_data))])
X_train= pd.DataFrame(generate_one_hot_list(train.Sequence))
Y_train = train.FitnessScore
X_val = pd.DataFrame(generate_one_hot_list(validate.Sequence))
Y_val = validate.FitnessScore
X_test=  pd.DataFrame(generate_one_hot_list(test.Sequence))
Y_test = test.FitnessScore

#%%
model = Sequential()

model.add(Dense(1218, activation='relu', input_shape=(1218,)))

model.add(Dense(1218, activation='relu'))
model.add(Dense(609, activation='relu'))
model.add(Dense(609, activation='relu'))

model.add(Dense(304, activation='relu'))
model.add(Dense(304, activation='relu'))
model.add(Dense(152, activation='relu'))

model.add(Dense(76, activation='relu'))

model.add(Dense(38, activation='relu'))

model.add(Dense(19, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(5, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr = 0.0001, decay =0.03), loss='mean_absolute_error', metrics=['mape'])
                
history12 = model.fit(X_train,Y_train, batch_size =100,epochs=40,verbose = 2, validation_split=0.2,  shuffle= True)
history_test = model.evaluate(X_test, Y_test, verbose =2)
model.summary()
Prediction = model.predict(X_test, verbose = 2)
Prediction_df = pd.DataFrame(Prediction)
Prediction_df.rename(columns={'0': 'Fitnessscore'})



#%%Histogram for scores
import seaborn as sns
sns.set()
plt.hist(df_affibody.df_unique.Score3/max(df.count()), bins = 10, color = 'maroon')
plt.grid()
plt.title('Affibody')
plt.yscale("log")
plt.ylabel('Frequency_log')
plt.xlabel('Score')
plt.style.use('ggplot')  
plt.show()
#%%
from scipy.stats import shapiro
a= shapiro(df_affibody.df_unique.Score3)
#%%
sns.distplot(df_affibody.df_unique.Score3, bins= 20)
plt.ylabel('Density')
plt.xlabel('Score')
plt.title('Affibody Score Distribution')
#%%Histogram for counts
counts = pd.concat([B1.df_unique, B2.df_unique, F1.df_unique, F2.df_unique])

fig = plt.figure(figsize=(15,8))

ax0 = fig.add_subplot(121)
ax0.hist(counts['count'], bins = [0,10,20,30,50,100], color = 'black')
# plt.xscale("log")
ax0.set_title(' Counts under 100', fontweight = 'bold')
ax0.set_xlabel('Score',fontweight = 'bold' , fontsize = 14)
ax0.set_ylabel('Frequency', fontweight = 'bold', fontsize = 14)
ax1 = fig.add_subplot(122)
ax1.hist(counts['count'], bins = [0,10,20,30,50,100,200,300,500,1000,5000,60000], color = 'purple')
ax1.set_title('log view of all counts', fontweight = 'bold', fontsize = 14)
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xlabel('Score_log',fontweight = 'bold', fontsize = 14)
ax1.set_ylabel('Frequency_log', fontweight = 'bold', fontsize = 14)
plt.style.use('ggplot') 
plt.tight_layout() 
plt.show()


#%% Plotting actual versus predicted value
import seaborn as sns
sns.jointplot(Prediction_df.actual_value,Prediction_df.predicted_value, kind='scatter', s=150, color='m', edgecolor="skyblue", linewidth=2)
# plt.xlim(0,1)
# plt.ylim(0,1)
plt.xscale("log")
plt.yscale("log")

#%%
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
Error = mape(Y_test, Prediction)
Y_test_df = pd.DataFrame(Y_test)
Y_test_df.reset_index(inplace = True)
Prediction_df['actual_value']= (Y_test_df.FitnessScore)
Prediction_df['predicted_value'] = Prediction
#%%CNN

X_train = pd.DataFrame(X_train).to_numpy()
X_test = pd.DataFrame(X_test).to_numpy()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#%%

#%%

model = Sequential()

model.add(Conv1D(100, 10, activation='relu', input_shape=(1218,1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(100, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(100, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(50, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

    
# model.add(Conv1D(50, kernel_size = 10, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))


model.add(Conv1D(50, kernel_size = 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dense(25, activation='relu'))
    
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr = 0.0001, decay = 0.003), loss='mean_absolute_error', metrics=['mape'])
                
history13= model.fit(X_train,Y_train,epochs=20, batch_size =100,verbose = 2, validation_split=0.2)
model.summary()#%% 
Prediction = model.predict(X_test, verbose = 2) 
#%% 
look_back = 1
n_features= X_train.shape[1]
X_train = X_train.values.reshape(4364, 1, 1218)
X_test=  X_test.values.reshape(1091, 1,1218)
#%%
model = Sequential()  
model.add(keras.layers.wrappers.Bidirectional(LSTM(100, input_shape=( look_back, 1218), return_sequences=True))) 

model.add(keras.layers.wrappers.Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.3))
model.add(keras.layers.wrappers.Bidirectional(LSTM(50)))
model.add(Dropout(0.2))
model.add((Dense(100, activation ='relu')))
model.add(Dropout(0.2))
model.add(Dense(50,  activation ='relu')) 
model.add(Dropout(0.2))
model.add(Dense(10,  activation ='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation ='relu'))   
model.compile(Adam(lr = 0.01, decay =0.005), loss='mean_absolute_error', metrics=['mape'])  
history8 = model.fit(X_train,Y_train, batch_size =20,epochs=20,verbose = 2, shuffle= True)

Prediction = model.predict(X_test, verbose = 2)
error = model.evaluate(X_test, Y_test)
#%%
#%%

plt.plot(history8.history['val_mape'], label = 'FeedForward, Square_root', color = 'blue')
plt.plot(history9.history['val_mape'], label = 'CNN, combining +Square_root', color = 'magenta')
plt.plot(history10.history['val_mape'], label = 'FeedForward, log2_Scoring', color = 'salmon')
plt.plot(history11.history['val_mape'], label ='CNN, log2_Scoring' , color = 'red')
plt.plot(history12.history['val_mape'], label = 'FeedForward, log10_Scoring', color = 'green')
plt.plot(history13.history['val_mape'], label = 'CNN, log10_Scoring', color = 'orange')
plt.title('model validation Eror')
plt.ylabel(' Validation Error')
plt.xlabel('epoch')
plt.legend()

# plt.plot(history.history['mape'], label = 'FeedForward, combining +Square_root', color = 'blue')
# plt.plot(history2.history['mape'], label = 'CNN, combining +Square_root', color = 'magenta')
# plt.plot(history3.history['mape'], label = 'FeedForward, log2_Scoring', color = 'black')



# plt.plot(history6.history['mape'], label ='CNN, log10_Scoring' , color = 'green')


plt.legend()
plt.axis([0,20,0,120])
 
  
