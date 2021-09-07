#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:00:52 2020

@author: zheng.1443
"""
#%% Initialize variables

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import time
import operator as op
from functools import reduce
import pickle
# from scipy.special import lambertw

K = 3 # total number of coins/arms/machines
M = 100 # number of experiments
N = 3000 # total budget

M_set = 50

epsilon = 0.05
delta = 1 - 0.95

theta = np.zeros((M_set,K))

for i in range(0,M_set):
    for j in range(0,K):
        theta[i,j] = np.random.uniform()
    
    
# print (theta)

#%%
theta_1 = 0.5
theta_2 = np.linspace(0.1,0.9,100)

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
#%% Lower Bound
        
n_Lower_Bound = np.zeros((theta.shape))

for i in range(0,M_set):
    for j in range(0,K):
        sigma_2 = theta[i,j]*(1-theta[i,j])
        n_Lower_Bound[i,j] = (sigma_2/(2*epsilon*epsilon))*(2*np.log(epsilon/(np.sqrt(2)*sigma_2*delta))-np.log(2*np.log(epsilon/(np.sqrt(2)*sigma_2*delta))))

#%% Offline Hoeffding

start_time = time.time()

# n_1_Hoeffding = N/K*np.ones((theta_2.shape))
# n_2_Hoeffding = N/K*np.ones((theta_2.shape))

n_Hoeffding = np.zeros((theta.shape))
n_Hoeffding.fill(K*np.log(2/delta)/(2*epsilon*epsilon))


# n_1_Hoeffding = K*np.log(2/delta)/(2*epsilon*epsilon)
# n_2_Hoeffding = K*np.log(2/delta)/(2*epsilon*epsilon)

# fig, ax = plt.subplots()
# ax.plot(theta_2, n_1_Hoeffding, label = 'n_1')
# ax.plot(theta_2, n_2_Hoeffding, label = 'n_2')
# legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

# plt.xlabel('theta_2')
# plt.ylabel('N_k')

# error_1_Hoeffding = np.zeros((theta_2.shape[0],M))
# error_2_Hoeffding = np.zeros((theta_2.shape[0],M))

# e1 = np.zeros((8,1))
# e2 = np.zeros((8,1))
# counter = 0
# for K in [1,2,5,10,20,50,100,1000]:
#     N=50*K
#     error_Hoeffding = np.zeros((M,K))
    
#     for i in range(0,M):
#         # for count,theta_head in enumerate(theta_2):
#         for count in range (0,K):
#             theta_1 = np.random.uniform(0,1)
#             flip_series = (np.random.rand(int(N/K)) <= theta_1).astype(int)    
#             error_Hoeffding[i,count] = abs(sum(flip_series)/(N/K) - theta_1)
#             # flip_series_1 = (np.random.rand(n_1_Hoeffding[count].astype(int)) <= theta_1).astype(int)
#             # flip_series_2 = (np.random.rand(n_2_Hoeffding[count].astype(int)) <= theta_head).astype(int)
            
#             # error_1_Hoeffding[count,i] = abs(sum(flip_series_1)/n_1_Hoeffding[count].astype(int) - theta_1)
#             # error_2_Hoeffding[count,i] = abs(sum(flip_series_2)/n_2_Hoeffding[count].astype(int) - theta_head)
#             # print(theta_1)
    
#     e1[counter] = np.average(np.amax(error_Hoeffding,axis=1),axis=0)
#     e2[counter] = np.amax(np.average(error_Hoeffding,axis=0),axis=0)
#     counter = counter+1
    

# fig, ax = plt.subplots()

# ax.plot([1,2,5,10,20,50,100,1000], e1, label = 'e_1')
# ax.plot([1,2,5,10,20,50,100,1000], e2, label = 'e_2')
# ax.set_xscale('log')
# legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

# plt.xlabel('log K')
# plt.ylabel('epsilon')
        
print("--- %s seconds ---" % (time.time() - start_time))

#%% Offline Bernstein
start_time = time.time()


# lambda_0=0
# n_1_Bernstein = np.ones((theta_2.shape))
# n_2_Bernstein = np.ones((theta_2.shape))

n_Bernstein = np.zeros((theta.shape))

# for count, theta_head in enumerate(theta_2):
#     epsilon_min = np.log(2/delta)*(2/3*K+np.sqrt(4/9*K*K+4*N*(2*(1+lambda_0)*theta_1*(1-theta_1)+2*(1+lambda_0)*theta_head*(1-theta_head))/np.log(2/delta)))/(2*N)

for i in range(0,M_set):
    for j in range(0,K):
        n_Bernstein[i,j] = np.log(2/delta)*(2*theta[i,j]*(1-theta[i,j])+2*epsilon/3)/(epsilon*epsilon)
        # n_Bernstein[i,j] = np.log(3/delta)*(2*theta[i,j]*(1-theta[i,j])+3*epsilon)/(epsilon*epsilon)

# n_1_Bernstein[count] = np.log(2/delta)*(2*theta[0]*(1-theta[0])+2*epsilon/3)/(epsilon*epsilon)
# n_2_Bernstein[count] = np.log(2/delta)*(2*theta[1]*(1-theta[1])+2*epsilon/3)/(epsilon*epsilon)
    
# fig, ax = plt.subplots()
# ax.plot(theta_2, n_1_Bernstein, label = 'n_1')
# ax.plot(theta_2, n_2_Bernstein, label = 'n_2')

# legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

# plt.xlabel('theta_2')
# plt.ylabel('N_k')
    
# error_1_Bernstein = np.zeros((theta_2.shape[0],M))
# error_2_Bernstein = np.zeros((theta_2.shape[0],M))

# for i in range(0,M):
#     for count,theta_head in enumerate(theta_2):
#         flip_series_1 = (np.random.rand(n_1_Bernstein[count].astype(int)) <= theta_1).astype(int)
#         flip_series_2 = (np.random.rand(n_2_Bernstein[count].astype(int)) <= theta_head).astype(int)
        
#         error_1_Bernstein[count,i] = abs(sum(flip_series_1)/n_1_Bernstein[count].astype(int) - theta_1)
#         error_2_Bernstein[count,i] = abs(sum(flip_series_2)/n_2_Bernstein[count].astype(int) - theta_head)

print("--- %s seconds ---" % (time.time() - start_time))
  
#%% Online Bernstein
start_time = time.time()

n_empirical_Bernstein = np.zeros((theta.shape))



for i in range(0,M_set):
    
    n_current = np.zeros((M,K))
    
    for j in range(0,M):
        counter=0
        n1_history = list()
        n2_history = list()
        n3_history = list() 
                    
        while (counter < N):    
            v1 = var_emp(n1_history)
            v2 = var_emp(n2_history)
            v3 = var_emp(n3_history)
                
            n_target_1 = np.log(3/delta)*(2*v1+3*epsilon)/(epsilon*epsilon)
            n_target_2 = np.log(3/delta)*(2*v2+3*epsilon)/(epsilon*epsilon)
            n_target_3 = np.log(3/delta)*(2*v3+3*epsilon)/(epsilon*epsilon)
        
            t = [[len(n1_history) < n_target_1],
                [len(n2_history) < n_target_2],
                [len(n3_history) < n_target_3]]           
            t = np.multiply(t,1)
            
            if (np.sum(t)==0):
                break
            
            t = t.tolist()
            flip = t.index(max(t))+1
            
            if flip == 1:
                if (np.random.uniform() <= theta[i,0]):
                    n1_history.append(1)
                else:
                    n1_history.append(0)
            elif flip == 2:
                if (np.random.uniform() <= theta[i,1]):
                    n2_history.append(1)
                else:
                    n2_history.append(0)
            elif flip == 3:
                if (np.random.uniform() <= theta[i,2]):
                    n3_history.append(1)
                else:
                    n3_history.append(0)
            
            counter=counter+1
        
        n_current[j,0]=len(n1_history)
        n_current[j,1]=len(n2_history)
        n_current[j,2]=len(n3_history)
    
    n_empirical_Bernstein[i,:] = np.average(n_current, axis=0)
    
    
    
    
print("--- %s seconds ---" % (time.time() - start_time))

#%% Online Bayesian
start_time = time.time()

n_empirical_Bayesian = np.zeros((theta.shape))



for i in range(0,M_set):
    
    n_current = np.zeros((M,K))
    
    for j in range(0,M):
        counter=0
        n1_history = list()
        n2_history = list()
        # n3_history = list() 
                    
        while (counter < N):    
            m1 = theta[i,0]#mean_emp(n1_history)
            m2 = theta[i,1]#mean_emp(n2_history)
            m3 = theta[i,2]#mean_emp(n3_history)
            
            d_target_1 = beta.cdf(m1+epsilon, sum(n1_history)+1, len(n1_history)-sum(n1_history)+1)-beta.cdf(m1-epsilon, sum(n1_history)+1, len(n1_history)-sum(n1_history)+1)
            d_target_2 = beta.cdf(m2+epsilon, sum(n2_history)+1, len(n2_history)-sum(n2_history)+1)-beta.cdf(m2-epsilon, sum(n2_history)+1, len(n2_history)-sum(n2_history)+1)
            d_target_3 = beta.cdf(m3+epsilon, sum(n3_history)+1, len(n3_history)-sum(n3_history)+1)-beta.cdf(m3-epsilon, sum(n3_history)+1, len(n3_history)-sum(n3_history)+1)


            t = [[delta < 1-d_target_1],
                [delta < 1-d_target_2],
                [delta < 1-d_target_3]]
            
            t = np.multiply(t,1)
            
            if (np.sum(t)==0):
                break
            
            t = t.tolist()
            flip = t.index(max(t))+1
            
            if flip == 1:
                if (np.random.uniform() <= theta[i,0]):
                    n1_history.append(1)
                else:
                    n1_history.append(0)
            elif flip == 2:
                if (np.random.uniform() <= theta[i,1]):
                    n2_history.append(1)
                else:
                    n2_history.append(0)
            elif flip == 3:
                if (np.random.uniform() <= theta[i,2]):
                    n3_history.append(1)
                else:
                    n3_history.append(0)
            
            counter=counter+1
        
        n_current[j,0]=len(n1_history)
        n_current[j,1]=len(n2_history)
        # n_current[j,2]=len(n3_history)
    
    n_empirical_Bayesian[i,:] = np.average(n_current, axis=0)
    
    
    
    
print("--- %s seconds ---" % (time.time() - start_time))
#%% plot distribution of budget

# n_1_naive = np.ones((theta_2.shape))
# n_2_naive = np.ones((theta_2.shape))

# for count, theta_head in enumerate(theta_2):
#     n_1_naive[count] = N*theta_1*(1-theta_1)/(theta_1*(1-theta_1)+theta_head*(1-theta_head))
#     n_2_naive[count] = N*theta_head*(1-theta_head)/(theta_1*(1-theta_1)+theta_head*(1-theta_head))
    

fig, ax = plt.subplots()

# ax.plot(theta_2, n_1_naive, label = 'n_1_naive')
# ax.plot(theta_2, n_2_naive, label = 'n_2_naive')
ax.plot(np.arange(1, M_set+1, 1), np.sum(n_Hoeffding,axis=1), label = 'n_Hoeffding')
ax.plot(np.arange(1, M_set+1, 1), np.sum(n_Bernstein,axis=1), label = 'n_Bernstein')
ax.plot(np.arange(1, M_set+1, 1), np.sum(n_empirical_Bernstein,axis=1), label = 'n_empirical_Bernstein')
# ax.plot(np.arange(1, M_set+1, 1), np.sum(n_empirical_Bayesian,axis=1), label = 'n_empirical_Bayesian')
# ax.plot(np.arange(1, M_set+1, 1), np.sum(n_Lower_Bound,axis=1), label = 'n_Lower_Bound')


# ax.plot(theta_2, n_1_Bernstein_graph, label = 'n_1_Bernstein_graph')
# ax.plot(theta_2, n_2_Bernstein_graph, label = 'n_2_Bernstein_graph')

# # ax.plot(theta_2, n_1_Hoeffding, label = 'n_1_Heoffding')

# ax.plot(theta_2, n_1_Empirical_average, label = 'n_1_Empirical_UCB')
# ax.plot(theta_2, n_2_Empirical_average, label = 'n_2_Empirical_UCB')

# ax.plot(theta_2, n_1_Empirical_var_average, label = 'n_1_Empirical')
# ax.plot(theta_2, n_2_Empirical_var_average, label = 'n_2_Empirical')


# ax.plot(theta_2, n_1_Bayesian_average, label = 'n_1_Bayesian')
# ax.plot(theta_2, n_2_Bayesian_average, label = 'n_2_Bayesian')

# for j in np.linspace(0,theta_2.shape[0]-1,10):
    # index = int(j)
    # ax.boxplot([n_1_Empirical_var[index,:],n_2_Empirical_var[index,:]],positions = [theta_2[index], theta_2[index]], whis=[2,98], showfliers=False, widths=0.03)
#     # ax.boxplot([n_1_Empirical[index,:],n_2_Empirical[index,:]],positions = [theta_2[index], theta_2[index]], whis=[2,98], showfliers=False, widths=0.03)



legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')
# plt.xlim(0,1)
from matplotlib.ticker import StrMethodFormatter
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.xlabel('theta_2')
plt.ylabel('N_k')
#%% Empirical UCB on variance
start_time = time.time()

n_1_Empirical = np.zeros((theta_2.shape[0],M))
n_2_Empirical = np.zeros((theta_2.shape[0],M))

error_1_Empirical = np.zeros((theta_2.shape[0],M))        
error_2_Empirical = np.zeros((theta_2.shape[0],M))

for i in range(0,M):
    
    for count, theta_head in enumerate(theta_2):
        
        n1_history = list()
        n2_history = list()       
                
        for j in range(0,N):
            if not(n1_history) or not(n2_history):
                if(np.random.uniform() >=0.5):
                    flip = 1
                else:
                    flip = 2
            elif var_UCB(j+1,n1_history) == var_UCB(j+1,n2_history):
                if(np.random.uniform() >=0.5):
                    flip = 1
                else:
                    flip = 2
            elif var_UCB(j+1,n1_history) > var_UCB(j+1,n2_history):
                flip = 1
            elif var_UCB(j+1,n1_history) < var_UCB(j+1,n2_history):
                flip = 2
                
            if flip == 1:
                n_1_Empirical[count,i] = n_1_Empirical[count,i]+1
                if (np.random.uniform() <= theta_1):
                    n1_history.append(1)
                else:
                    n1_history.append(0)
                
            elif flip == 2:
                n_2_Empirical[count,i] = n_2_Empirical[count,i]+1
                if (np.random.uniform() <= theta_head):
                    n2_history.append(1)
                else:
                    n2_history.append(0)
            
        error_1_Empirical[count,i] = abs(sum(n1_history)/n_1_Empirical[count,i] - theta_1)
        error_2_Empirical[count,i] = abs(sum(n2_history)/n_2_Empirical[count,i] - theta_head)


n_1_Empirical_average = np.average(n_1_Empirical, axis=1)
n_2_Empirical_average = np.average(n_2_Empirical, axis=1)


print("--- %s seconds ---" % (time.time() - start_time))

#%% Empirical on confidence interval
start_time = time.time()

n_1_Empirical_var = np.zeros((theta_2.shape[0],M))
n_2_Empirical_var = np.zeros((theta_2.shape[0],M))

error_1_Empirical_var = np.zeros((theta_2.shape[0],M))        
error_2_Empirical_var = np.zeros((theta_2.shape[0],M))


for i in range(0,M):
    
    for count, theta_head in enumerate(theta_2):
        
        n1_history = list()
        n2_history = list()       
                
        for j in range(0,N):           
            if var_emp(n1_history) == var_emp(n2_history):
                if(np.random.uniform() >=0.5):
                    flip = 1
                else:
                    flip = 2
            elif var_emp(n1_history) > var_emp(n2_history):
                flip = 1
            elif var_emp(n1_history) < var_emp(n2_history):
                flip = 2
                
            if flip == 1:
                n_1_Empirical_var[count,i] = n_1_Empirical_var[count,i]+1
                if (np.random.uniform() <= theta_1):
                    n1_history.append(1)
                else:
                    n1_history.append(0)
                
            elif flip == 2:
                n_2_Empirical_var[count,i] = n_2_Empirical_var[count,i]+1
                if (np.random.uniform() <= theta_head):
                    n2_history.append(1)
                else:
                    n2_history.append(0)
            
        error_1_Empirical_var[count,i] = abs(sum(n1_history)/n_1_Empirical_var[count,i] - theta_1)
        error_2_Empirical_var[count,i] = abs(sum(n2_history)/n_2_Empirical_var[count,i] - theta_head)


n_1_Empirical_var_average = np.average(n_1_Empirical_var, axis=1)
n_2_Empirical_var_average = np.average(n_2_Empirical_var, axis=1)


print("--- %s seconds ---" % (time.time() - start_time))
    
#%% Bayesian + Greedy
start_time = time.time()

n_1_Bayesian = np.zeros((theta_2.shape[0],M))
n_2_Bayesian = np.zeros((theta_2.shape[0],M))

error_1_Bayesian = np.zeros((theta_2.shape[0],M))
error_2_Bayesian = np.zeros((theta_2.shape[0],M))

for i in range(0,M):   
        
    for count, theta_head in enumerate(theta_2):
        
        a_1 = 1
        b_1 = 1
    
        a_2 = 1
        b_2 = 1
        
        for j in range(0,N):
        
            if beta.var(a_1,b_1) == beta.var(a_2,b_2):
                if(np.random.uniform() >=0.5):
                    flip = 1
                else:
                    flip = 2
            elif beta.var(a_1,b_1) > beta.var(a_2,b_2):
                flip = 1
            else:
                flip = 2
            
            if flip == 1:
                n_1_Bayesian[count,i] = n_1_Bayesian[count,i]+1
                if (np.random.uniform() <= theta_1):
                    a_1 = a_1+1
                else:
                    b_1 = b_1+1
            elif flip == 2:
                n_2_Bayesian[count,i] = n_2_Bayesian[count,i]+1
                if (np.random.uniform() <= theta_head):
                    a_2 = a_2+1
                else:
                    b_2 = b_2+1
        
        error_1_Bayesian[count,i] = abs((a_1-1)/(a_1+b_1-2) - theta_1)
        error_2_Bayesian[count,i] = abs((a_2-1)/(a_2+b_2-2) - theta_head)
        
n_1_Bayesian_average = np.average(n_1_Bayesian,axis=1)
n_2_Bayesian_average = np.average(n_2_Bayesian,axis=1)               

print("--- %s seconds ---" % (time.time() - start_time))

#%%
with open('Empirical_400.pkl', 'rb') as file:
    n_1_Empirical, n_2_Empirical, n_1_Empirical_average, n_2_Empirical_average,theta_2 = pickle.load(file)



#%%

with open('Empirical_800.pkl','wb') as file:
    pickle.dump([n_1_Empirical, n_2_Empirical, n_1_Empirical_average, n_2_Empirical_average,theta_2],file)

#%%
# import pickle

# with open('Hoedffding_2M.pkl','wb') as file:
#     pickle.dump([n_1_Hoeffding, error_1_Hoeffding,error_2_Hoeffding, theta_2],file)

# with open('Hoedffding_2M.pkl', 'rb') as file:
#     n_1_Hoeffding, error_1_Hoeffding,error_2_Hoeffding, theta_2 = pickle.load(file)
    
error_1_Hoeffding_theory = np.zeros((theta_2.shape[0],1))
error_2_Hoeffding_theory = np.zeros((theta_2.shape[0],1))
 

for count,theta_head in enumerate(theta_2):           
    # error_1_Hoeffding_theory[count] =  mean_abs_dev(n_1_Hoeffding[count],theta_1)
    # error_2_Hoeffding_theory[count] = mean_abs_dev(n_2_Hoeffding[count],theta_head)
    error_1_Hoeffding_theory[count] =  np.log(2/delta)*(2/3+np.sqrt(4/9+4*(N/K)*(2*theta_1*(1-theta_1))/np.log(2/delta)))/(2*N/K)
    error_2_Hoeffding_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(N/K)*(2*theta_head*(1-theta_head))/np.log(2/delta)))/(2*N/K)
                                                       
#%% Bernstein with lambda parameter
start_time = time.time()

lambda_range = [-1,-0.99,-0.90, -0.7, 0, 2, 10, 100, 1000000]

fig, ax = plt.subplots()
# ax.plot(theta_2, n_1_Hoeffding, label = 'n_1_Heoffding')

# error_1_Bernstein_theory = np.zeros((lambda_range.shape[0],1))
# error_2_Bernstein_theory = np.zeros((lambda_range.shape[0],1))

for count, lambda_0 in enumerate(lambda_range):
# lambda_0 = 0
    n_1_Bernstein = np.ones((theta_2.shape))
    n_2_Bernstein = np.ones((theta_2.shape))
    
    for count, theta_head in enumerate(theta_2):
        epsilon_min = np.log(2/delta)*(2/3*K+np.sqrt(4/9*K*K+4*N*(2*(1+lambda_0)*theta_1*(1-theta_1)+2*(1+lambda_0)*theta_head*(1-theta_head))/np.log(2/delta)))/(2*N)
        
        n_1_Bernstein[count] = np.log(2/delta)*(2*(1+lambda_0)*theta_1*(1-theta_1)+2/3*epsilon_min)/(epsilon_min*epsilon_min)
        n_2_Bernstein[count] = np.log(2/delta)*(2*(1+lambda_0)*theta_head*(1-theta_head)+2/3*epsilon_min)/(epsilon_min*epsilon_min)
        
    # error_1_Bernstein_theory = np.zeros((theta_2.shape[0],1))
    # error_2_Bernstein_theory = np.zeros((theta_2.shape[0],1))

    # for count,theta_head in enumerate(theta_2):      
    #     error_1_Bernstein_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_1_Bernstein[count])*(2*theta_1*(1-theta_1))/np.log(2/delta)))/(2*n_1_Bernstein[count])
    #     error_2_Bernstein_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_2_Bernstein[count])*(2*theta_head*(1-theta_head))/np.log(2/delta)))/(2*n_2_Bernstein[count])


    # ax.plot(theta_2, np.maximum(error_1_Bernstein_theory,error_2_Bernstein_theory), label = str(lambda_0))
    ax.plot(theta_2, n_1_Bernstein, label = 'n_1_Bernstein_'+str(lambda_0))
    ax.plot(theta_2, n_2_Bernstein, label = 'n_2_Bernstein_'+str(lambda_0))

# ax.plot(lambda_range, np.maximum(error_1_Bernstein_theory, error_2_Bernstein_theory))
legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

plt.xlabel('theta_2')
plt.ylabel('N_k')



print("--- %s seconds ---" % (time.time() - start_time))
    
#%% single coin flip
start_time = time.time()

error_single_coin = np.zeros((theta_2.shape[0],M))
epsilon_coin = np.zeros((theta_2.shape[0],M))

for i in range(0,M):
    for count,theta_head in enumerate(theta_2):
        flip_series = (np.random.rand(N) <= theta_head).astype(int)     
        
        epsilon_coin[count,i] = np.log(2/delta)*(1+np.sqrt(1+18*N*theta_head*(1-theta_head)/np.log(2/delta)))/(3*N)
        error_single_coin[count,i] = abs(sum(flip_series)/N - theta_head)

error_single_coin_average = np.average(error_single_coin, axis=1)  
epsilon_coin_average = np.average(epsilon_coin, axis=1)    

fig, ax = plt.subplots()

ax.plot(theta_2, epsilon_coin_average, label = 'epsilon_coin')
ax.plot(theta_2, error_single_coin_average, label = 'error_single_coin')


for j in np.linspace(0,theta_2.shape[0]-1,10):
    index = int(j)
    ax.boxplot([error_single_coin[index,:]],positions = [theta_2[index]], whis=[2.5,97.5], showfliers=False, widths=0.03)
    # ax.boxplot([n_1_Empirical[index,:],n_2_Empirical[index,:]],positions = [theta_2[index], theta_2[index]], whis=[2,98], showfliers=False, widths=0.03)


legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')
plt.xlim(0,1)
from matplotlib.ticker import StrMethodFormatter
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.xlabel('theta_2')
plt.ylabel('epsilon')

print("--- %s seconds ---" % (time.time() - start_time)) 

#%%
    
# with open('Bernstein_2M.pkl','wb') as file:
#     pickle.dump([n_1_Bernstein, n_2_Bernstein, error_1_Bernstein,error_2_Bernstein, theta_2],file)
    
# with open('Bernstein_2M.pkl', 'rb') as file:
#     n_1_Bernstein, n_2_Bernstein, error_1_Bernstein, error_2_Bernstein, theta_2 = pickle.load(file)
    


error_1_Bernstein_theory = np.zeros((theta_2.shape[0],1))
error_2_Bernstein_theory = np.zeros((theta_2.shape[0],1))

for count,theta_head in enumerate(theta_2):      
    error_1_Bernstein_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4/K*(N/K)*(2*theta_1*(1-theta_1)+2*theta_head*(1-theta_head))/np.log(2/delta)))/(2*N/K)
    error_2_Bernstein_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4/K*(N/K)*(2*theta_1*(1-theta_1)+2*theta_head*(1-theta_head))/np.log(2/delta)))/(2*N/K)

#%%
    
error_1_Empirical_theory = np.zeros((theta_2.shape[0],1))
error_2_Empirical_theory = np.zeros((theta_2.shape[0],1))

for count,theta_head in enumerate(theta_2):      
    error_1_Empirical_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_1_Empirical_average[count])*(2*theta_1*(1-theta_1))/np.log(2/delta)))/(2*n_1_Empirical_average[count])
    error_2_Empirical_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_2_Empirical_average[count])*(2*theta_head*(1-theta_head))/np.log(2/delta)))/(2*n_2_Empirical_average[count])

#%%
    
error_1_Bayesian_theory = np.zeros((theta_2.shape[0],1))
error_2_Bayesian_theory = np.zeros((theta_2.shape[0],1))

for count,theta_head in enumerate(theta_2):      
    error_1_Bayesian_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_1_Bayesian_average[count])*(2*theta_1*(1-theta_1))/np.log(2/delta)))/(2*n_1_Bayesian_average[count])
    error_2_Bayesian_theory[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_2_Bayesian_average[count])*(2*theta_head*(1-theta_head))/np.log(2/delta)))/(2*n_2_Bayesian_average[count])

#%%

error_1_naive = np.zeros((theta_2.shape[0],1))
error_2_naive  = np.zeros((theta_2.shape[0],1))

for count,theta_head in enumerate(theta_2):      
    error_1_naive[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_1_naive[count])*(2*theta_1*(1-theta_1))/np.log(2/delta)))/(2*n_1_naive[count])
    error_2_naive[count] = np.log(2/delta)*(2/3+np.sqrt(4/9+4*(n_2_naive[count])*(2*theta_head*(1-theta_head))/np.log(2/delta)))/(2*n_2_naive[count])



#%% plot minimax errors
    
e1 = np.average(np.maximum(error_1_Bernstein,error_2_Bernstein),axis=1)

e2 = np.maximum(np.average(error_1_Bernstein,axis=1),np.average(error_2_Bernstein,axis=1))

# e3 = np.maximum(np.quantile(error_1_Empirical,0.95,axis=1),np.quantile(error_2_Empirical,0.95,axis=1))

# e4 = np.maximum(np.quantile(error_1_Bayesian,0.95,axis=1),np.quantile(error_2_Bayesian,0.95,axis=1))


# e0 = np.maximum(error_1_naive,error_2_naive)

# e5 = np.maximum(error_1_Hoeffding_theory,error_2_Hoeffding_theory)

# e6 = np.maximum(error_1_Bernstein_theory,error_2_Bernstein_theory)

# e7 = np.maximum(error_1_Empirical_theory,error_2_Empirical_theory)

# e8 = np.maximum(error_1_Bayesian_theory,error_2_Bayesian_theory)

fig, ax = plt.subplots()

# ax.plot(theta_2, e1/e2, label = 'ratio')

# ax.plot(theta_2, e3/e4, label = 'ratio_theory')

# ax.plot(theta_2, e1, label = 'e_Hoeffding')

# ax.plot(theta_2, e2, label = 'e_Bernstein')

# ax.plot(theta_2, e3, label = 'e_Empirical')

# ax.plot(theta_2, e4, label = 'e_Bayesian')



ax.plot(theta_2, e1, label = 'e_E_max')

ax.plot(theta_2, e2, label = 'e_max_E')

# ax.plot(theta_2, e7, label = 'e_Empirical Theory')

# ax.plot(theta_2, e8, label = 'e_Bayesian Theory')

# ax.plot(theta_2, e0, label = 'e_naive Theory')

# ax.plot(theta_2, np.average(np.maximum(error_1_Empirical, error_2_Empirical),axis=1), label = 'e_Empirical')

legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

plt.xlabel('theta_2')
plt.ylabel('epsilon')

#%% KL Bound method numerical solution

def KL_Bernoulli (p1,p2):
    return p1*np.log(p1/p2)+(1-p1)*np.log((1-p1)/(1-p2))


def KL_method (theta_test):   

    epsilon_min = np.zeros((theta_test.shape[0],1))
    n1_min = np.zeros((theta_test.shape[0],1))
    n2_min = np.zeros((theta_test.shape[0],1))
    
    for count_1,theta_0 in enumerate(theta_test):
        
        # epsilon_test = np.linspace(0.001,min(0.999-theta_0,theta_0-0.001),100)
        epsilon_test = np.linspace(0.001,0.3,1000)
        # KL_plus = np.zeros((1000,1))
        # KL_minus = np.zeros((1000,1))
        N_total = np.zeros((1000,1))
        
        for count_2, epsilon_0 in enumerate(epsilon_test):
            # KL_plus[count_2] = KL_Bernoulli(theta_0+epsilon_0, theta_0)
            # KL_minus[count_2] = KL_Bernoulli(theta_0-epsilon_0,theta_0)
            
            if theta_0 >= 0.5:
                KL_2 = KL_Bernoulli(theta_0-epsilon_0, theta_0)
            else:
                KL_2 = KL_Bernoulli(theta_0+epsilon_0, theta_0)
            
            N_total[count_2] = np.log(2/delta)/KL_2 + np.log(2/delta)/KL_Bernoulli(theta_1+epsilon_0,theta_1)
            
            if N_total[count_2]<N:
                epsilon_min[count_1] = epsilon_0
                n1_min[count_1] = np.log(2/delta)/KL_Bernoulli(theta_1+epsilon_0,theta_1)
                n2_min[count_1] = np.log(2/delta)/KL_2
                break
    
    return (epsilon_min,n1_min,n2_min)
 



        
fig, ax = plt.subplots()

# ax.plot(theta_2, e5, label = 'e_Hoeffding Theory')

# ax.plot(theta_2, e6, label = 'e_Bernstein Theory')

# ax.plot(theta_2, e1, label = 'e_Hoeffding')

# ax.plot(theta_2, e2, label = 'e_Bernstein')

# ax.plot(theta_2, e3, label = 'e_Empirical')

# ax.plot(theta_2, e4, label = 'e_Bayesian')

# ax.plot(theta_test, epsilon_min, label = 'epsilon_min')


# legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

# plt.xlabel('theta_test')
# plt.ylabel('n')


# theta_test_1 = np.linspace(0.5,0.9,500)
# (epsilon_min_1,n1_min,n2_min) = KL_method (theta_test_1)

# ax.plot(theta_test_1, epsilon_min_1, label = 'epsilon_min')

# ax.plot(theta_test_1, n1_min, label = 'n1')
# ax.plot(theta_test_1, n2_min, label = 'n2')

# theta_test_2 = np.linspace(0.1,0.5,500)   
# (epsilon_min_2,n1_min,n2_min) = KL_method (theta_test_2)

# ax.plot(theta_test_2, epsilon_min_2, label = 'epsilon_min')


ax.plot(theta_test, n1_min, label = 'n1')
ax.plot(theta_test, n2_min, label = 'n2')

ax.plot(theta_2, n_1_naive, label = 'n_1_naive')
ax.plot(theta_2, n_2_naive, label = 'n_2_naive')

ax.plot(theta_2, n_1_Bernstein, label = 'n_1_Bernstein')
ax.plot(theta_2, n_2_Bernstein, label = 'n_2_Bernstein')

legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

plt.xlabel('theta_test')
plt.ylabel('n')

#%% Bionomial CDF numerical solution

from scipy.stats import binom
import math

start_time = time.time()

theta_test = np.linspace(0.1,0.9,100)

def CDF_method (theta_test):   

    epsilon_min = np.zeros((theta_test.shape[0],1))
    n1_min = np.zeros((theta_test.shape[0],1))
    n2_min = np.zeros((theta_test.shape[0],1))
    
    for count_1,theta_0 in enumerate(theta_test):
        print(theta_0)
        # epsilon_test = np.linspace(0.001,min(0.999-theta_0,theta_0-0.001),100)
        epsilon_test = np.linspace(0.05,0.15,100)
        # KL_plus = np.zeros((1000,1))
        # KL_minus = np.zeros((1000,1))
        N_total = np.zeros((1000,1))
        
        for count_2, epsilon_0 in enumerate(epsilon_test):
            for n1 in np.linspace(0,N,N+1):
                k1 = math.floor(n1*max(theta_1-epsilon_0,0))
                k2 = math.ceil(n1*min(theta_1+epsilon_0,1))
                
                e_prob_1 = binom.cdf(k1,n1,theta_1) + binom.cdf(n1-k2,n1,1-theta_1)
                
                n2 = int(N-n1)
                k1 = math.floor(n2*max(theta_0-epsilon_0,0))
                k2 = math.ceil(n2*min(theta_0+epsilon_0,1))
                
                e_prob_2 = binom.cdf(k1,n2,theta_0) + binom.cdf(n2-k2,n2,1-theta_0)
                
                # print(e_prob_1,e_prob_2)
                if(e_prob_1 < delta and e_prob_2 < delta):
                    epsilon_min[count_1] = epsilon_0
                    n1_min[count_1] = n1
                    n2_min[count_1] = n2
                    break
            
            if epsilon_min[count_1]>0:
                break
                
    
    return (epsilon_min,n1_min,n2_min)

epsilon_min,n1_min,n2_min = CDF_method(theta_test)
 
print("--- %s seconds ---" % (time.time() - start_time))
