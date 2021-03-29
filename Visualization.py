"""
This file provides some visualizartion guide for making comparison between different set ups and visualize the data distribution in different platforms

"""



#for visualizing the data distribution distplot in seaborn is an appropriate function and can be impelemnted such as the following
import searborn as sns
sns.set()
N = 20 # depends on the data
sns.distplot(Fitness_Score, bins= N)
plt.ylabel('Density')
plt.xlabel('Score')
plt.title('My_Protein Score Distribution')




#%%


#In order to get some understanding of the predicted results and model performance, the scatterplot seeking for 1-to-1 ratio is an appropriate choice
#1-to-1 ratio refers to predicted vs. actual values

import seaborn as sns
sns.jointplot(Actual,Predicted kind='scatter', s=150, color='m', edgecolor="skyblue", linewidth=2)
plt.xscale("log")
plt.yscale("log")
plt.ylabel('Predicted')
plt.xlabel('Actual')

#%%


# One other important visualization is to track the loss function VS. the epoch number

# Storing the history of fitting and loss function value by defining a variable for model.fit

History = model.fit(X_train,Y_train, batch_size =100,epochs=20,verbose = 2, shuffle= True)
plt.plot(History)