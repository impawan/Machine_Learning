# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:46:23 2018

@author: paprasad
"""

#import os 
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt


path = 'D:\\Users\\paprasad\\python\\Source\\LRSRC.txt'


data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#data.head() 
#data.describe()  

data.insert(0, 'Ones', 1)
#print (data)
#print (data.shape[0])
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))  
cols = data.shape[1]  
row = data.shape[0]
#print (row)
#print (cols)
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols] 

X = np.matrix(X.values)  
y = np.matrix(y.values) 



theta = np.matrix(np.array([0,0]))



cost = computeCost(X, y, theta)


alpha = 0.01
iters = 1000

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)  


print (g)
final_cost = computeCost(X,y,g)

print(final_cost)




x = np.linspace(data.Population.min(), data.Population.max(), 100)  
f = g[0, 0] + (g[0, 1] * x)


fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size') 


fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  


def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)     
    return np.sum(inner) / (2 * len(X))




def gradinetDecsent(X,y,theta,iters):
    temp=np.matrix(np.zeros(theta.shape))
    paremeters = int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    print (temp)
    for i in range(iters):
        error = (X*theta.T)-y
        for j in range(paremeters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            theta = temp
            cost[i] = computeCost(X, y, theta)
        return theta, cost   
        
    
    
    

