#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:40:18 2019

@author: zheng.1443
"""

import sys
import glob
import numpy as np
import torch
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
import time
import math
import random
import scipy.io as sio
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import pickle
import math
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.axes.Axes.plot
#matplotlib.pyplot.plot
#matplotlib.axes.Axes.legend
#matplotlib.pyplot.legend

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator

#%%

#A = np.array([[0,0,0,1],
#              [0,0,0,1],
#              [0,0,0,1],
#              [0,0,0,0]])

A = np.array([[0,0,1],
              [0,0,1],
              [0,0,0]
              ])

N = A.shape[0]
#p0 = 0.3
p1 = 0.7
#p2 = 0.5

#data = {'1': [],
#        '2': [],
#        '3': []}


T=20
K=20
n_sim=0
S_final = np.zeros((T,N,K))

while (n_sim < K):
    S_0 = np.ones(N, dtype=int)
    S_0[np.random.choice(N,1)] = 0 #random initial failure
    
    S_t = np.array(S_0)
    S_all = np.array(S_0)
        
    counter = 1
    while (counter < T): 
        S_t_new = np.array(S_t)
        for i in range(0,N):
            if (S_t[i] == 1):
                n_node = np.where(A[:,i]==1)
                fail_node = np.where(S_t==0)
                n_fail = np.intersect1d(n_node,fail_node).size
                if (n_fail>0):
                    p_fail = 1-pow((1-p1),n_fail)
                    t = np.random.uniform()
    #                print(i,p_fail,t)
                    if(t<p_fail):
                        S_t_new[i]=0
              
        S_t = np.array(S_t_new)
        S_all = np.vstack((S_all, S_t))
#        data['1'].append(S_t[0])
#        data['2'].append(S_t[1])
#        data['3'].append(S_t[2])
        counter = counter+1
    
    S_final[:,:,n_sim] = S_all
    n_sim = n_sim+1
#print(data)
#%%

def get_likelihood(S,A, theta):
    total_time = S.shape[0]
    N_node = S.shape[1]
    
    total_likelihood = torch.zeros(1)
    for i in range(0,total_time-1):
        y_0 = S[i,:]
        y_1 = S[i+1,:]
        
        likelihood = torch.zeros(1)        
        for j in range(0,N_node):
            n_node = np.where(A[:,j]==1)
            fail_node = np.where(y_0==0)
            n_fail = np.intersect1d(n_node,fail_node).size
            
            if(y_1[j]==0 and y_0[j] == 0):
                likelihood = torch.ones(1) 
            elif(y_1[j]==0 and y_0[j] == 1):
                if n_fail > 0:
                    likelihood = 1-pow((1-theta),n_fail)
            elif(y_1[j]==1 and y_0[j] == 1):
                if n_fail == 0:
                    likelihood = torch.ones(1)
                elif n_fail > 0:
                    likelihood = pow((1-theta),n_fail)

            total_likelihood = total_likelihood+torch.log(likelihood)
    
    return total_likelihood
#%%


theta = torch.rand(1,requires_grad = True)

for epoch in range(0,50):
    ll = torch.zeros(1)
    for i in range(0,K):
        S = S_final[:,:,i]
        ll = ll+ get_likelihood(S, A, theta)    
       
    ll.backward()
    
#    print (theta.grad) 
    with torch.no_grad():
        theta = theta + 0.001* theta.grad
                     

    theta.requires_grad = True
    print (theta)

#%%

#import torch
torch.manual_seed(0)
N = 100
x = torch.rand(N,1)*5
# Let the following command be the true function
y = 2.3 + 5.1*x
# Get some noisy observations
y_obs = y + 0.2*torch.randn(N,1)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


gamma = 0.01
for i in range(500):
    print(i)
    # use new weight to calculate loss
    y_pred = w * x + b
    mse = torch.mean((y_pred - y_obs) ** 2)

    # backward
    mse.backward()
    print('w:', w)
    print('b:', b)
    print('w.grad:', w.grad)
    print('b.grad:', b.grad)


#%%
data = pd.DataFrame(data)

model = BayesianModel([('1', '3'), ('2', '3')]) 

#pe = ParameterEstimator(model, data)
#print("\n", pe.state_counts('1'))
#print("\n", pe.state_counts('3'))  

mle = MaximumLikelihoodEstimator(model, data)
print(mle.estimate_cpd('1'))  # unconditional
print(mle.estimate_cpd('3'))  # conditional

model.fit(data, estimator=MaximumLikelihoodEstimator)