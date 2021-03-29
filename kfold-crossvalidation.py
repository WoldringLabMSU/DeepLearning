
import keras
import pandas as pd
from keras.models import Sequential
from functions_mehrsa import generate_onehot_list
from keras.layers.core import Dense
import numpy as np
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.model_selection import KFold

#%%
df= pd.read_csv('my_protein.csv', index_col=False)


#%%
fold_no = 5
acc_per_fold = []
loss_per_fold = []
inputs = np.array(generate_onehot_list(df.Sequence))
targets = df.Fitness_Score

kf = KFold(fold_no, shuffle=True, random_state=42)
    
for train, test in kf.split(inputs,targets):
    

    model = Sequential()

    model.add(Dense(1218, activation='relu', input_shape=(1218,)))

    model.add(Dense(609, activation='relu'))
    model.add(Dense(304, activation='relu'))
    model.add(Dense(152, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(76, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(19, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1,  activation='relu'))
    model.compile(Adam(lr = 0.001), loss='mean_squared_error', metrics=['mape'])
                
    model.fit(inputs[train], targets[train],  batch_size =85,epochs=300,verbose = 2, shuffle= True)
    model.summary()
    Prediction = model.predict(inputs[test], verbose = 2)
    print(f'Training for fold {fold_no} ...')
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}%')
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

  # Increase fold number
    fold_no = fold_no + 1
    
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - mape: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> mape: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
       
#%%
