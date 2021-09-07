# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:56:17 2020

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

K = 10 # total number of nodes
M = 100 # number of experiments to take average
N_max = 8000 # total budget

epsilon = 0.05
delta = 1 - 0.95

theta = np.zeros((K))
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
        
theta = np.zeros((K))

# for i in range(0,K):
#     # theta[i] = np.random.beta(0.5,0.5) #low variance
#     theta[i] = np.random.beta(1,1) #uniform
#     # theta[i] = np.random.beta(2,2) #high variance
    
# # print (theta)

#%% Lower Bound
        
# n_Lower_Bound = np.zeros((theta.shape))

# for i in range(0,K):
#     sigma_sq = theta[i]*(1-theta[i])
#     temp = epsilon/(np.sqrt(2)*sigma_sq*delta)
#     n_Lower_Bound[i] = (sigma_sq/(2*epsilon*epsilon))*(2*np.log(temp)-np.log(2*np.log(temp)))

# for i in range(1,K):
#     n_Lower_Bound[i] = n_Lower_Bound[i-1]+n_Lower_Bound[i]

#%% Offline Hoeffding

n_Hoeffding = np.zeros((theta.shape))
n_Hoeffding.fill(np.log(2/delta)/(2*epsilon*epsilon))

for i in range(1,K):
    n_Hoeffding[i] = n_Hoeffding[i]+n_Hoeffding[i-1]

#%% Low Variance
    
# Offline Bernstein
n_Bernstein_L = np.zeros((theta.shape))

for i in range(0,K):
    theta[i] = np.random.beta(0.5,0.5) #low variance

for i in range(0,K):
    n_Bernstein_L[i] = np.log(2/delta)*(2*theta[i]*(1-theta[i])+2*epsilon/3)/(epsilon*epsilon)
    
for i in range(1,K):
    n_Bernstein_L[i] = n_Bernstein_L[i]+n_Bernstein_L[i-1]
    

#% Online Bernstein
start_time = time.time()

n_empirical_Bernstein_L = np.zeros((theta.shape))

   
n_current = np.zeros((M,K))

for i in range(0,M): #running M times to take average
    counter=0
    
    n_history = [[] for x in range(K)]    
                
    while (counter < N_max):
        
        variance_emp = np.zeros((theta.shape))
        n_target = np.zeros((theta.shape))
        r = np.zeros((theta.shape))
        
        for j in range(0,K):            
            variance_emp[j] = var_emp(n_history[j])
            n_target[j] = np.log(3/delta)*(2*variance_emp[j]+3*epsilon)/(epsilon*epsilon)
            r[j] = len(n_history[j]) < n_target[j]
    
            
        r = np.multiply(r,1)
        
        if (np.sum(r)==0):
            break
        
        r = r.tolist()
        flip = r.index(max(r))
        
       
        if (np.random.uniform() <= theta[flip]):
            n_history[flip].append(1)
        else:
            n_history[flip].append(0)      
          
        counter=counter+1
    
    for j in range(0,K):
        n_current[i,j]=len(n_history[j])
    print("--- %s seconds ---" % (time.time() - start_time))

n_empirical_Bernstein_L = np.average(n_current, axis=0)

for i in range(1,K):
    n_empirical_Bernstein_L[i] = n_empirical_Bernstein_L[i]+n_empirical_Bernstein_L[i-1]

print("--- %s seconds ---" % (time.time() - start_time))



#%% Uniform
   
for i in range(0,K):
    theta[i] = np.random.beta(1,1) #uniform

n_Bernstein_U = np.zeros((theta.shape))

for i in range(0,K):
    n_Bernstein_U[i] = np.log(2/delta)*(2*theta[i]*(1-theta[i])+2*epsilon/3)/(epsilon*epsilon)
    
for i in range(1,K):
    n_Bernstein_U[i] = n_Bernstein_U[i]+n_Bernstein_U[i-1]
    
#% Online Bernstein
start_time = time.time()

n_empirical_Bernstein_U = np.zeros((theta.shape))

   
n_current = np.zeros((M,K))

for i in range(0,M): #running M times to take average
    counter=0
    
    n_history = [[] for x in range(K)]    
                
    while (counter < N_max):
        
        variance_emp = np.zeros((theta.shape))
        n_target = np.zeros((theta.shape))
        r = np.zeros((theta.shape))
        
        for j in range(0,K):            
            variance_emp[j] = var_emp(n_history[j])
            n_target[j] = np.log(3/delta)*(2*variance_emp[j]+3*epsilon)/(epsilon*epsilon)
            r[j] = len(n_history[j]) < n_target[j]
    
            
        r = np.multiply(r,1)
        
        if (np.sum(r)==0):
            break
        
        r = r.tolist()
        flip = r.index(max(r))
        
       
        if (np.random.uniform() <= theta[flip]):
            n_history[flip].append(1)
        else:
            n_history[flip].append(0)      
          
        counter=counter+1
    
    for j in range(0,K):
        n_current[i,j]=len(n_history[j])
    print("--- %s seconds ---" % (time.time() - start_time))

n_empirical_Bernstein_U = np.average(n_current, axis=0)

for i in range(1,K):
    n_empirical_Bernstein_U[i] = n_empirical_Bernstein_U[i]+n_empirical_Bernstein_U[i-1]

print("--- %s seconds ---" % (time.time() - start_time))

    
#%% High variance

for i in range(0,K):
    theta[i] = np.random.beta(2,2) #high variance

n_Bernstein_H = np.zeros((theta.shape))

for i in range(0,K):
    n_Bernstein_H[i] = np.log(2/delta)*(2*theta[i]*(1-theta[i])+2*epsilon/3)/(epsilon*epsilon)
    
for i in range(1,K):
    n_Bernstein_H[i] = n_Bernstein_H[i]+n_Bernstein_H[i-1]


#%Online Bernstein
start_time = time.time()

n_empirical_Bernstein_H = np.zeros((theta.shape))

   
n_current = np.zeros((M,K))

for i in range(0,M): #running M times to take average
    counter=0
    
    n_history = [[] for x in range(K)]    
                
    while (counter < N_max):
        
        variance_emp = np.zeros((theta.shape))
        n_target = np.zeros((theta.shape))
        r = np.zeros((theta.shape))
        
        for j in range(0,K):            
            variance_emp[j] = var_emp(n_history[j])
            n_target[j] = np.log(3/delta)*(2*variance_emp[j]+3*epsilon)/(epsilon*epsilon)
            r[j] = len(n_history[j]) < n_target[j]
    
            
        r = np.multiply(r,1)
        
        if (np.sum(r)==0):
            break
        
        r = r.tolist()
        flip = r.index(max(r))
        n_current[i,flip] = n_current[i,flip]+1
       
        if (np.random.uniform() <= theta[flip]):
            n_history[flip].append(1)
        else:
            n_history[flip].append(0)      
          
        counter=counter+1
    
  
    print("--- %s seconds ---" % (time.time() - start_time))

n_empirical_Bernstein_H = np.average(n_current, axis=0)

for i in range(1,K):
    n_empirical_Bernstein_H[i] = n_empirical_Bernstein_H[i]+n_empirical_Bernstein_H[i-1]

print("--- %s seconds ---" % (time.time() - start_time))


#%%  
fig, ax = plt.subplots()

# ax.plot(theta_2, n_1_naive, label = 'n_1_naive')
# ax.plot(theta_2, n_2_naive, label = 'n_2_naive')
ax.plot(np.arange(200, K+1, 1), n_Hoeffding[199:K+1], label = 'n_Hoeffding')
ax.plot(np.arange(200, K+1, 1), n_Bernstein_L[199:K+1], label = 'n_Bernstein_L')
ax.plot(np.arange(200, K+1, 1), n_Bernstein_U[199:K+1], label = 'n_Bernstein_U')
ax.plot(np.arange(200, K+1, 1), n_Bernstein_H[199:K+1], label = 'n_Bernstein_H')
# ax.plot(np.arange(1, K+1, 1), n_empirical_Bernstein_L, label = 'n_empirical_Bernstein_L')
# ax.plot(np.arange(1, K+1, 1), n_empirical_Bernstein_U, label = 'n_empirical_Bernstein_U')
# ax.plot(np.arange(1, K+1, 1), n_empirical_Bernstein_H, label = 'n_empirical_Bernstein_H')

# ax.plot(np.arange(1, K+1, 1), n_Lower_Bound, label = 'n_Lower_Bound')
# ax.set_yscale('log')

legend = ax.legend(loc='upper left', shadow=False, fontsize='x-small')

# from matplotlib.ticker import StrMethodFormatter
# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.xlabel('Number of Nodes (K)')
plt.ylabel('Toal Sample Complexity (N)')