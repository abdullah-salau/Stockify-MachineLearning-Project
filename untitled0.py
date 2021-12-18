# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:35:57 2021

@author: abdul
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn import model_selection
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

arr = [None,2,3,4,5,None,7,8,9,None]
arr_copy = arr
arr_copy.remove(9)
print(arr_copy)
drop = [2,3,4]
arr = [x for x in arr if x not in drop]
print(arr)

series = pd.Series(arr)
print(series.isnull().sum()/series.size)

data=pd.DataFrame({'Month':['January','April','March','April','Februay','June','July','June','September']})

import category_encoders as ce
hash_enc = ce.HashingEncoder(cols='Month', n_components=3)
hash_enc_data = hash_enc.fit_transform(data)
#inputDF = hash_enc_data
print(data)
print(hash_enc_data)