# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:29:50 2018

@author: impawan
"""

import numpy as np
import pandas as pd

from sklearn  import preprocessing

#setting path varaible for source file 
path ='D:\Machine Learning\src\HeightWeightAgeGender.txt'

np.seterr(divide='ignore')

#creating dataframe from source file using pandas lib
data = pd.read_csv(path,sep=';',skiprows=1,names=['Height','Weight','Age','Male'])
zipped = list(zip(data.Height,data.Male,data.Age,data.Weight))
modidata = pd.DataFrame(zipped)

scalar = preprocessing.MinMaxScaler()
#
#modidata = scalar.fit_transform(tempdf)
#modidata = pd.DataFrame(modidata)


print(type (modidata))
#print (modidata)

#ax = modidata.plot(kind='scatter',x=0,y=1,figsize=(12,8))
alpha = 0.001
iteration = 1000


def isnan(num):
    return num!=num


def CostFunction(X,Y,Theta):
    SqrError = np.power((X*Theta.T-Y),2)
    cost = .5*np.sum(SqrError)/len(X)
    return cost

def GradeintDes (X,Y,Theta):
    ParamtersCnt = int(Theta.ravel().shape[1])
    temp = np.matrix(np.zeros(Theta.shape))
    cost = np.zeros(iteration)
    for i in range(iteration):
        Error = (X*Theta.T-Y)
        for j in range(ParamtersCnt):
            term = np.multiply(Error,X[:,j])
            temp[0,j]=temp[0,j]-(alpha/len(X))*np.sum(term)
            Theta = temp
        cost[i]=CostFunction(X,Y,Theta)
        print (cost[i]," ",i)
        if(isnan(cost[i])):
            return
    return Theta,cost            
         

def LinearRegression(df) :
    #df.insert(0,'Ones',1)
    col = df.shape[1];
    x=df.iloc[:,0:col-1]
    y=df.iloc[:,col-1:col]
    X= np.matrix(x.values)
    Y= np.matrix(y.values)
    X=scalar.fit_transform(X)
    X.insert(0,'ones',1)
    print (X)
    Theta = np.matrix(np.zeros(X.shape[1]))
    print(Theta.shape)
    print (X.shape)
    g,cost = GradeintDes(X,Y,Theta)
    final_cost = CostFunction(X,Y,g)
    print ("The parameters are ",g)
    print ("Value of Cost is ",final_cost)
    
LinearRegression(modidata)

    