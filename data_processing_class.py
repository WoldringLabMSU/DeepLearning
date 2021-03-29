#Importing the required libraries

import pandas as pd
import numpy as np
import os, sys, math
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import scipy.stats
from scipy.stats import boxcox
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

#%%

class sequence(): # defining a class called sequnece in order to store all the attributes and impelement different fucntions
    
    def __init__(self,name):
        self.name = name
        
    def read_seq(self,name_of_file,path_of_file):
        """
        In this function by defining the path which data files are loacated and the file names, their information will be stored
        """
        self.file = name_of_file
        self.path = path_of_file
        self.df = pd.read_csv(self.path +"\\" +self.file)
        
    
    def find_unique(self):
        
        """
        This function is used in order to calculate the number of times each sequence has been shown as a binder in the provided deep sequencing files
        
        """
        
        self.df['count'] = 1
        self.df_unique = self.df.groupby("Sequence").agg({'count':'sum'}).reset_index() # Make sure to conver the FASTA to csv and name the sequence column as "Sequence"
        
    def remove_underline(self):
        """
        This part is for cleaning the data from an undesired letter or sign in the sequences. Underline in here is an example to show the code for getting rid of undesired
        signs and letters in input file
        """
        self.df_unique = self.df_unique[~self.df_unique['Sequence'].str.contains("_")]
        
      
    def remove_freq_below_limit(self, freq_limit): #freq_limit determines the cut-off value for removing the bad binders
        self.limit = freq_limit
        above_limit = self.df_unique['count'] >self.limit
        self.df_filtered = self.df_unique[above_limit] # the filtered dataset will be the new data set to work with which has met the defined criteria
    
       
    def add_dataframe(self, data):
           
        self.df_unique = pd.concat([self.df_unique, data])
        
       #The dataset can be normalized or standardized but it needs some preprocessing to perform well
       
    def normalize_data_set(self):
        
        """
        This function is used for normalizing the dataset, the sklearn preprocessing can be used as well(sklearn.preprocessing.normalize())
        """
        self.df_filtered['count'] = (self.df_filtered['count']-min(self.df_filtered['count'])) /(max(self.df_filtered['count'])-min(self.df_filtered['count']))
        
        
        
    def standardize_data_set(self):
        
        scaler1 = StandardScaler() # this will be more toward the normal distribution
        scaler1.fit((self.df_unique['count']))
        
        self.df_filtered['count'] = scaler1.transform(self.df_filtered['count'])
        
        
    def standardize2_data_set(self):
        # this is useful for reducing the influence of outliers 
        scaler2 = RobustScaler()
        scaler2.fit(self.df_filtered['count'])
        self.df_filtered['count'] = scaler2.transform(self.df_filtered['count'])
        
        
        
        
    def generate_onehot_list(self):
        
        """
        This function is used for one-hot encoding the sequences by the use of amino acid dictionary
        """
        
        AA = ["A","C", "D", "E","F", "G", "H", "I", "K", "L", "M", "N", "P","Q","R", "S", "T", "V","W", "Y", "-"]
        one_hot_length = len(AA)
        one_hots=[]

        for i in range(one_hot_length):
            a = np.zeros(one_hot_length)
            a[i] = 1
            one_hots.append(a)
   
        one_hot_dict = dict(zip(AA,one_hots))


        seq_one_hot_list =[]

        for seq in self.df_unique.Sequence:
            seq_one_hot =[]
            for item in seq:
                seq_one_hot.append(one_hot_dict[item])
            seq_one_hot = np.concatenate(seq_one_hot).ravel()
            seq_one_hot_list.append(seq_one_hot)
            
        self.one_hot = pd.DataFrame(seq_one_hot_list)
        
        
    def  generate_onehot_list2(self):
        
        """
        Rather than manually creating the one-hot encoded sequences, the user can get the help of sklearn preprocessing package
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder

        Amino_Acids = ["A","C", "D", "E","F", "G", "H", "I", "K", "L", "M", "N", "P","Q","R", "S", "T", "V","W", "Y"]
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse = False)
        integer_encode = label_encoder.fit_transform(Amino_Acids)
        integer_encoded = integer_encode.reshape(len(integer_encode), 1)
        Amino_Acids_onehot = onehot_encoder.fit_transform(integer_encoded)
        self.one_hot = pd.DataFrame(Amino_Acids_onehot)






#The method for processing the y value ( label) depend on the specific appliacation.  The four following functions are steps for defining the score function ( fine-tuned and processed values from raw frequencies)
    def normalize_within_library(self,other):
        
        """
        This is an arbitrary function which will be useful dealing with different populations. Therefore, dome normalization is required in order to enable comparison 
        between the two populations with that of different sample numbers.
        
        An example for two population would be the time when there are two different sets of data from FACS and MACS are available
        
        """
        
        self.normalization_factor = (len(self.df)/ len(other.df))
        other.df_unique['count'] = other.df_unique['count']*self.normalization_factor
        
        
    def get_common_unique_filtered_seq(self,other): 
        # when two populations have some common sequences and each sequence has its own frequency it each file, the common sequences should be treated separately
        self.df_common = self.df_filtered.merge(other.df_filtered, on = 'Sequence')
        self.df_common.rename (columns = {"count_x": "count_" + str(self.name), "count_y": "count_" + str(other.name)}, inplace = True)
        
    def get_exclusive_unique_filtered_seq(self, other):
        
        
        self.df_exclusive = pd.concat([self.df_filtered, other.df_filtered]).drop_duplicates(subset= 'Sequence', keep=False, inplace=False)  #when keep is equal to false, drop_duplicate command will provide all the sequences which are unique to each poulation
        
        
    def get_score1(self,other):
        
        """
        This calculates the score function, population-based. Iy takes the average score between the common sequences in data set and calculates square root for all the frequencies.
        
        """
        self.df_common['FitnessScore'] = (self.df_common["count_" + str(self.name)]**0.5 + self.df_common["count_" + str(other.name)]**0.5)/2
        self.df_exclusive["FitnessScore"] = self.df_exclusive['count']**0.5
        self.df_all = pd.concat([self.df_exclusive, self.df_common])

   
        
    
    def get_score2(self):
        #taking the log is among the common power transform functions for changing the distribution
       
       self.df_unique['Score2'] = (np.log10( list(self.df_unique['count'])))
       
    def get_score3(self):
        #Box-Cox transform
        BoxCox= list(scipy.stats.boxcox(list(self.df_unique['count'])))
        self.df_unique['Score3'] =BoxCox[0]
        
        
    def get_score4(self):
     # Yeo-jphnson transform
        Yeojphnson= list(scipy.stats.yeojohnson(list(self.df_unique['count'])))
        self.df_unique['Score4'] =Yeojphnson[0]
         
    
    

            
