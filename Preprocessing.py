# -*- coding: utf-8 -*-
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
from matplotlib import pyplot


class sequence():
    
    def __init__(self,name):
        self.name = name
        
    def read_seq(self,name_of_file,path_of_file):
        
        self.file = name_of_file
        self.path = path_of_file
        self.df = pd.read_csv(self.path +"\\" +self.file)
        
    
    def find_unique(self):
        
        self.df['count'] = 1
        self.df_unique = self.df.groupby("Sequence").agg({'count':'sum'}).reset_index()
        
    def remove_underline(self):
        self.df_unique = self.df_unique[~self.df_unique['Sequence'].str.contains("_")]
       
    def normalize_within_library(self,other):
        
        self.normalization_factor = (len(self.df)/ len(other.df))
        other.df_unique['count'] = other.df_unique['count']*self.normalization_factor
        
    def normalize_data_set(self):
        self.df_unique['count'] = (self.df_unique['count']-min(self.df_unique['count'])) /(max(self.df_unique['count'])-min(self.df_unique['count']))
        
    def remove_freq_below_limit(self, freq_limit):
        self.limit = freq_limit
        above_limit = self.df_unique['count'] >self.limit
        self.df_filtered = self.df_unique[above_limit]
    
    def get_common_unique_filtered_seq(self,other):
        self.df_common = self.df_filtered.merge(other.df_filtered, on = 'Sequence')
        self.df_common.rename (columns = {"count_x": "count_" + str(self.name), "count_y": "count_" + str(other.name)}, inplace = True)
        
    def get_exclusive_unique_filtered_seq(self, other):
        
        self.df_exclusive = pd.concat([self.df_filtered, other.df_filtered]).drop_duplicates(subset= 'Sequence', keep=False, inplace=False)  
        
    def get_score1(self,other):
        self.df_common['FitnessScore'] = (self.df_common["count_" + str(self.name)]**0.5 + self.df_common["count_" + str(other.name)]**0.5)/2
        self.df_exclusive["FitnessScore"] = self.df_exclusive['count']**0.5
        self.df_all = pd.concat([self.df_exclusive, self.df_common])

    def generate_onehot_list1(self):
        
        codes = ["A","C", "D", "E","F", "G", "H", "I", "K", "L", "M", "N", "P","Q","R", "S", "T", "V","W", "Y", "-"]
        one_hot_length = len(codes)
        one_hots=[]

        for i in range(one_hot_length):
            a = np.zeros(one_hot_length)
            a[i] = 1
            one_hots.append(a)
   
        one_hot_dict = dict(zip(codes,one_hots))


        seq_one_hot_list =[]

        for seq in self.df_all.Sequence:
            seq_one_hot =[]
            for item in seq:
                seq_one_hot.append(one_hot_dict[item])
            seq_one_hot = np.concatenate(seq_one_hot).ravel()
            seq_one_hot_list.append(seq_one_hot)
            
        self.one_hot = pd.DataFrame(seq_one_hot_list)
        
        
        
    def generate_onehot_list2(self):
        
        codes = ["A","C", "D", "E","F", "G", "H", "I", "K", "L", "M", "N", "P","Q","R", "S", "T", "V","W", "Y", "-"]
        one_hot_length = len(codes)
        one_hots=[]

        for i in range(one_hot_length):
            a = np.zeros(one_hot_length)
            a[i] = 1
            one_hots.append(a)
   
        one_hot_dict = dict(zip(codes,one_hots))


        seq_one_hot_list =[]

        for seq in self.df_unique.Sequence:
            seq_one_hot =[]
            for item in seq:
                seq_one_hot.append(one_hot_dict[item])
            seq_one_hot = np.concatenate(seq_one_hot).ravel()
            seq_one_hot_list.append(seq_one_hot)
            
        self.one_hot = pd.DataFrame(seq_one_hot_list)
        
        
    
    def get_score2(self, a=0.00005, b=0, c=30, d=10, e= 50, f=400):
    
        y=[]
        for item in list(self.df_unique['count']):
            y.append(a*item +b+ c/(1+(np.exp(f*(1/(item+d)**0.5-1/e)))))
        self.df_unique['Score2'] = y
        
    def get_score3(self):
       
       self.df_unique['Score3'] = (np.log10( list(self.df_unique['count'])))
       
    def get_score4(self):
        #Box-Cox transform
        a= list(scipy.stats.boxcox(list(self.df_unique['count'])))
        self.df_unique['Score4'] =a[0]
        
        
    def get_score5(self):
     # Yeo-jphnson transform
         b= list(scipy.stats.yeojohnson(list(self.df_unique['count'])))
         self.df_unique['Score5'] =b[0]
         
    
    def add_dataframe(self, data):
           
        self.df_unique = pd.concat([self.df_unique, data])
        

            
