# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:56:39 2020

@author: Yilin Zheng
"""

#%% Initialize variables

import numpy as np
import matplotlib.pyplot as plt
import time
import operator as op
from functools import reduce
import pickle
from scipy.special import lambertw
from scipy.stats import beta

K = 5 # total number of nodes
M = 100 # number of experiments to take average
N_max = 4000 # total budget

epsilon = 0.05
delta = 1 - 0.95

theta = np.ones(K)*0.3#np.random.uniform()

#%% utility functions

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def mean_abs_dev (n,p):
    k = np.floor(n*p)+1
    return (2*k*ncr(int(n),int(k))*np.power(p,k)*np.power(1-p,n-k+1))/n

def var_UCB (n,n_history):
    alpha = 3.01   
    return np.var(np.asarray(n_history)) + (7*alpha*np.log(n)/(6*len(n_history))+np.sqrt(alpha*np.log(n)/(2*len(n_history))))

def var_emp (n_history):
    # n_len = len(n_history)
    if not(n_history):
        return float("inf")
    else:        
        # return np.sqrt(2*np.var(np.asarray(n_history))*np.log(3/delta)/n_len) + 3*np.log(3/delta)/n_len
        return np.var(n_history)

def mean_emp (n_history):
    # n_len = len(n_history)
    if not(n_history):
        return 0.5
    else:        
        # return np.sqrt(2*np.var(np.asarray(n_history))*np.log(3/delta)/n_len) + 3*np.log(3/delta)/n_len
        return np.mean(n_history)

def CI_emp (n_history):
    n_len = len(n_history)
    if not(n_history):
        return float("inf")
    else:        
        return np.sqrt(2*np.var(np.asarray(n_history))*np.log(3/delta)/n_len) + 3*np.log(3/delta)/n_len

#%%
        
tree_structure = np.array([[1,0,theta[0],0,0],
                                  [0,1,theta[1],0,0],
                                  [0,0,1,theta[2],theta[2]],
                                  [0,0,0,1,0],
                                  [0,0,0,0,1]])

#%% New algorithm
start_time = time.time()

n_empirical_Bernstein_New = np.zeros((theta.shape))  
n_current = np.zeros((M,K))

for i in range(0,M):
    
    counter=0
        
    n_history = [[] for x in range(K)] 
                
    while (counter < N_max): 
        
        variance_emp = np.zeros((theta.shape))
        n_target = np.zeros((theta.shape))
        r = np.zeros((theta.shape))
        
        
        tree_structure = np.array([[1,0,theta[0],0,0],
                              [0,1,theta[1],0,0],
                              [0,0,1,theta[2],theta[2]],
                              [0,0,0,1,0],
                              [0,0,0,0,1]])
        
        for j in range(0,K):            
           variance_emp[j] = var_emp(n_history[j])
           # n_target[j] = np.log(3/delta)*(2*variance_emp[j]+3*epsilon)/(epsilon*epsilon) #empirical Bernstein
           n_target[j] = np.log(2/delta)*(2*theta[j]*(1-theta[j])+2/3*epsilon)/(epsilon*epsilon) #empirical Bernstein           
           r[j] = len(n_history[j]) < n_target[j]
 
        r = np.multiply(r,1)
        
        if (np.sum(r)==0):
            break
        
        p_hat = np.dot(tree_structure,r)
        p_hat = p_hat.tolist()
        flip = p_hat.index(max(p_hat))
        n_current[i,flip] = n_current[i,flip]+1
           
        if (np.random.uniform() <= theta[flip]):
            n_history[flip].append(1)
            
            unlearned_child = np.multiply(tree_structure[flip,:],r)
            for cc,value in enumerate(unlearned_child):               
                if value > 0 and value <1:
                    if (np.random.uniform() <= theta[cc]):
                        n_history[cc].append(1)
                    else:
                        n_history[cc].append(0)            
        else:
            n_history[flip].append(0)
        
        counter = counter+1
 
    print("--- %s seconds ---" % (time.time() - start_time))    
n_empirical_Bernstein_New = np.average(n_current, axis=0)

print("--- %s seconds ---" % (time.time() - start_time))