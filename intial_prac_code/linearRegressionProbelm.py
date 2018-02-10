# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:29:50 2018

@author: ACER
"""

import numpy as py
import pandas as pd

#setting path varaible for source file 
path ='D:\Machine Learning\src\HeightWeightAgeGender.txt'


data = pd.read_csv(path,sep=';',skiprows=1)

print(type(data))


