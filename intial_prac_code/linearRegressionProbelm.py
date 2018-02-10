# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:29:50 2018

@author: impawan
"""

import numpy as np
import pandas as pd

#setting path varaible for source file 
path ='D:\Machine Learning\src\HeightWeightAgeGender.txt'

#creating dataframe from source file using pandas lib
data = pd.read_csv(path,sep=';',skiprows=1,names=['Height','Weight','Age','Male'])


zipped = list(zip(data.Height,data.Male,data.Age,data.Weight))
modidata = pd.DataFrame(zipped)

#print (modidata)

#ax = modidata.plot(kind='scatter',x=0,y=1,figsize=(12,8))
aplha =0.001;
iteration = 1000


def LinearRegression(df,alpha,iteration) :
    df.insert(0,'Ones',1)
    col = df.shape[1];
    x=df.iloc[:,0:col-1]
    y=df.iloc[:,col-1:col]
    X= np.matrix(x.values)
    Y= np.matrix(y.values)
    Theta = np.matrix(np.zeros(df.shape[0]))
    print (Theta.shape)

LinearRegression(modidata,aplha,iteration)

    