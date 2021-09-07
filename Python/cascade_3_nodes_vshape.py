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
# import pickle

M = 100 # number of experiments
N = 1000 # total budget

K = 3 # total number of coins/arms/machines

epsilon = 0.05
delta = 1 - 0.95

theta_1 = np.linspace(0.1,0.9,100)

theta_2 = 0.5

theta_3 = 0.5

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
    n_len = len(n_history)
    if not(n_history):
        return float("inf")
    else:        
        return np.sqrt(2*np.var(np.asarray(n_history))*np.log(3/delta)/n_len) + 3*np.log(3/delta)/n_len
    
#%% Bernstein 2-node graph

n_1_Bernstein_graph = np.ones((theta_1.shape))
n_2_Bernstein_graph = np.ones((theta_1.shape))
n_3_Bernstein_graph = np.ones((theta_1.shape))

for count, theta_head in enumerate(theta_1):
    
    alpha_1 = 1-theta_head
    alpha_2 = 1-theta_2
    alpha_3 = 1
    
    sigma_1 = (1-theta_head)*theta_head
    sigma_2 = (1-theta_2)*theta_2
    sigma_3 = (1-theta_3)*theta_3
    
    c_prime = N/((alpha_1+alpha_2+alpha_3)*np.log(2/delta))    
    epsilon_min = (1+np.sqrt(1+18*c_prime*(alpha_1*sigma_1+alpha_2*sigma_2+alpha_3*sigma_3)/(alpha_1+alpha_2+alpha_3)))/(3*c_prime) 
    n_1_Bernstein_graph[count] = np.log(2/delta)*(2*sigma_1+2/3*epsilon_min)/(epsilon_min*epsilon_min)
    n_2_Bernstein_graph[count] = np.log(2/delta)*(2*sigma_2+2/3*epsilon_min)/(epsilon_min*epsilon_min)
    n_3_Bernstein_graph[count] = np.log(2/delta)*(2*sigma_3+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph[count]*theta_head-n_2_Bernstein_graph[count]*theta_2
  
   
    
    if n_3_Bernstein_graph[count]<0:
        alpha_1=1
        alpha_2=1
        alpha_3=0
        c_prime = N/((alpha_1+alpha_2+alpha_3)*np.log(2/delta))    
        epsilon_min = (1+np.sqrt(1+18*c_prime*(alpha_1*sigma_1+alpha_2*sigma_2+alpha_3*sigma_3)/(alpha_1+alpha_2+alpha_3)))/(3*c_prime) 
        n_1_Bernstein_graph[count] = np.log(2/delta)*(2*sigma_1+2/3*epsilon_min)/(epsilon_min*epsilon_min)
        n_2_Bernstein_graph[count] = np.log(2/delta)*(2*sigma_2+2/3*epsilon_min)/(epsilon_min*epsilon_min)
        n_3_Bernstein_graph[count] = 0      
 
        

#%% Bernstein
# start_time = time.time()


# lambda_0=0
# n_1_Bernstein = np.ones((theta_2.shape))
# n_2_Bernstein = np.ones((theta_2.shape))

# for count, theta_head in enumerate(theta_2):
#     epsilon_min = np.log(2/delta)*(2/3*K+np.sqrt(4/9*K*K+4*N*(2*(1+lambda_0)*theta_1*(1-theta_1)+2*(1+lambda_0)*theta_head*(1-theta_head))/np.log(2/delta)))/(2*N)
    
#     n_1_Bernstein[count] = np.log(2/delta)*(2*(1+lambda_0)*theta_1*(1-theta_1)+2/3*epsilon_min)/(epsilon_min*epsilon_min)
#     n_2_Bernstein[count] = np.log(2/delta)*(2*(1+lambda_0)*theta_head*(1-theta_head)+2/3*epsilon_min)/(epsilon_min*epsilon_min)
        
   
# error_1_Bernstein = np.zeros((theta_2.shape[0],M))
# error_2_Bernstein = np.zeros((theta_2.shape[0],M))

# # for i in range(0,M):
# #     for count,theta_head in enumerate(theta_2):
# #         flip_series_1 = (np.random.rand(n_1_Bernstein[count].astype(int)) <= theta_1).astype(int)
# #         flip_series_2 = (np.random.rand(n_2_Bernstein[count].astype(int)) <= theta_head).astype(int)
        
# #         error_1_Bernstein[count,i] = abs(sum(flip_series_1)/n_1_Bernstein[count].astype(int) - theta_1)
# #         error_2_Bernstein[count,i] = abs(sum(flip_series_2)/n_2_Bernstein[count].astype(int) - theta_head)

# print("--- %s seconds ---" % (time.time() - start_time))     

        

#%% Bayesian + Greedy
start_time = time.time()

n_1_Bayesian = np.zeros((theta_1.shape[0],M))
n_2_Bayesian = np.zeros((theta_1.shape[0],M))
n_3_Bayesian = np.zeros((theta_1.shape[0],M))

# error_1_Bayesian = np.zeros((theta_2.shape[0],M))
# error_2_Bayesian = np.zeros((theta_2.shape[0],M))

for i in range(0,M):   
        
    for count, theta_head in enumerate(theta_1):
        
        a_1 = 1
        b_1 = 1
    
        a_2 = 1
        b_2 = 1
        
        a_3 = 1
        b_3 = 1
        
        for j in range(0,N):
        
            t1 = beta.var(a_1,b_1)
            t2 = beta.var(a_2,b_2)
            t3 = beta.var(a_3,b_3)
            
            t=[t1,t2,t3]            
            flip = t.index(max(t))+1
 
            if flip == 1:
                n_1_Bayesian[count,i] = n_1_Bayesian[count,i]+1
                if (np.random.uniform() <= theta_head):
                    a_1 = a_1+1                  
                    
                    if (np.random.uniform() <= theta_3):
                        a_3 = a_3+1
                    else:
                        b_3 = b_3+1               
                   
                else:
                    b_1 = b_1+1
                    
            elif flip == 2:
                n_2_Bayesian[count,i] = n_2_Bayesian[count,i]+1
                if (np.random.uniform() <= theta_2):
                    a_2 = a_2+1
                    
                    if (np.random.uniform() <= theta_3):
                        a_3 = a_3+1
                    else:
                        b_3 = b_3+1
                else:
                    b_2 = b_2+1
            
            elif flip == 3:
                n_3_Bayesian[count,i] = n_3_Bayesian[count,i]+1
                if (np.random.uniform() <= theta_3):
                    a_3 = a_3+1
                else:
                    b_3 = b_3+1
        
        # error_1_Bayesian[count,i] = abs((a_1-1)/(a_1+b_1-2) - theta_1)
        # error_2_Bayesian[count,i] = abs((a_2-1)/(a_2+b_2-2) - theta_head)
        
n_1_Bayesian_average = np.average(n_1_Bayesian,axis=1)
n_2_Bayesian_average = np.average(n_2_Bayesian,axis=1)
n_3_Bayesian_average = np.average(n_3_Bayesian,axis=1)               
               

print("--- %s seconds ---" % (time.time() - start_time))




#%% Empirical on confidence interval
start_time = time.time()

n_1_Empirical_var = np.zeros((theta_1.shape[0],M))
n_2_Empirical_var = np.zeros((theta_1.shape[0],M))
n_3_Empirical_var = np.zeros((theta_1.shape[0],M))


# error_1_Empirical_var = np.zeros((theta_2.shape[0],M))        
# error_2_Empirical_var = np.zeros((theta_2.shape[0],M))


for i in range(0,M):
    
    for count, theta_head in enumerate(theta_1):
        
        n1_history = list()
        n2_history = list()
        n3_history = list() 
                
        for j in range(0,N): 
            
            t1 = var_emp(n1_history)
            t2 = var_emp(n2_history)
            t3 = var_emp(n3_history)
            
            t=[t1,t2,t3]            
            flip = t.index(max(t))+1           
                           
            if flip == 1:
                n_1_Empirical_var[count,i] = n_1_Empirical_var[count,i]+1
                if (np.random.uniform() <= theta_head):
                    n1_history.append(1)
                    
                    if (np.random.uniform() <= theta_2):
                        n2_history.append(1)
                        
                        if (np.random.uniform() <= theta_3):
                            n3_history.append(1)
                        else:
                            n3_history.append(0)  
                        
                    else:
                        n2_history.append(0)
                        
                   
                else:
                    n1_history.append(0)
                
            elif flip == 2:
                n_2_Empirical_var[count,i] = n_2_Empirical_var[count,i]+1
                if (np.random.uniform() <= theta_2):
                    n2_history.append(1)
                    
                    if (np.random.uniform() <= theta_3):
                        n3_history.append(1)
                    else:
                        n3_history.append(0)  
                else:
                    n2_history.append(0)
                    
            elif flip == 3:
                n_3_Empirical_var[count,i] = n_3_Empirical_var[count,i]+1
                if (np.random.uniform() <= theta_3):
                    n3_history.append(1)
                else:
                    n3_history.append(0)
            
        # error_1_Empirical_var[count,i] = abs(sum(n1_history)/n_1_Empirical_var[count,i] - theta_1)
        # error_2_Empirical_var[count,i] = abs(sum(n2_history)/n_2_Empirical_var[count,i] - theta_head)


n_1_Empirical_var_average = np.average(n_1_Empirical_var, axis=1)
n_2_Empirical_var_average = np.average(n_2_Empirical_var, axis=1)
n_3_Empirical_var_average = np.average(n_3_Empirical_var, axis=1)



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

# ax.plot(theta_2, n_1_Bernstein, label = 'n_1_Bernstein')
# ax.plot(theta_2, n_2_Bernstein, label = 'n_2_Bernstein')

ax.plot(theta_1, n_1_Bernstein_graph, label = 'n_1_Bernstein_graph')
ax.plot(theta_1, n_2_Bernstein_graph, label = 'n_2_Bernstein_graph')
ax.plot(theta_1, n_3_Bernstein_graph, label = 'n_3_Bernstein_graph')


# ax.plot(theta_1, n_1_Empirical_var_average, label = 'n_1_Empirical')
# ax.plot(theta_1, n_2_Empirical_var_average, label = 'n_2_Empirical')
# ax.plot(theta_1, n_3_Empirical_var_average, label = 'n_3_Empirical')


# ax.plot(theta_2, n_1_Empirical_graph_average, label = 'n_1_Empirical_g')
# ax.plot(theta_2, n_2_Empirical_graph_average, label = 'n_2_Empirical_g')


ax.plot(theta_1, n_1_Bayesian_average, label = 'n_1_Bayesian')
ax.plot(theta_1, n_2_Bayesian_average, label = 'n_2_Bayesian')
ax.plot(theta_1, n_3_Bayesian_average, label = 'n_3_Bayesian')

# ax.plot(theta_2, n_1_Bayesian_graph_average, label = 'n_1_Bayesian_g')
# ax.plot(theta_2, n_2_Bayesian_graph_average, label = 'n_2_Bayesian_g')

# for j in np.linspace(0,theta_2.shape[0]-1,10):
    # index = int(j)
    # ax.boxplot([n_1_Empirical_var[index,:],n_2_Empirical_var[index,:]],positions = [theta_2[index], theta_2[index]], whis=[2,98], showfliers=False, widths=0.03)
#     # ax.boxplot([n_1_Empirical[index,:],n_2_Empirical[index,:]],positions = [theta_2[index], theta_2[index]], whis=[2,98], showfliers=False, widths=0.03)



legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')
plt.xlim(0,1)
from matplotlib.ticker import StrMethodFormatter
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.xlabel('theta_1')
plt.ylabel('N_k')

#%% Bayesian 2-node graph
# start_time = time.time()

# n_1_Bayesian_graph = np.zeros((theta_2.shape[0],M))
# n_2_Bayesian_graph = np.zeros((theta_2.shape[0],M))

# error_1_Bayesian = np.zeros((theta_2.shape[0],M))
# error_2_Bayesian = np.zeros((theta_2.shape[0],M))

# for i in range(0,M):   
        
#     for count, theta_head in enumerate(theta_2):
        
#         a_1 = 1
#         b_1 = 1
    
#         a_2 = 1
#         b_2 = 1
        
#         for j in range(0,N):
                        
#             t2 = theta_head*(beta.var(a_2,b_2)-beta.var(a_2+1,b_2))+(1-theta_head)*(beta.var(a_2,b_2)-beta.var(a_2,b_2+1))
#             t1 = theta_1*(beta.var(a_1,b_1)-beta.var(a_1+1,b_1)+t2)+(1-theta_1)*(beta.var(a_1,b_1)-beta.var(a_1,b_1+1))
            
#             if t1 == t2:
#                 if(np.random.uniform() >=0.5):
#                     flip = 1
#                 else:
#                     flip = 2
#             elif t1 > t2:
#                 flip = 1
#             else:
#                 flip = 2
            
#             if flip == 1:
#                 n_1_Bayesian_graph[count,i] = n_1_Bayesian_graph[count,i]+1
#                 if (np.random.uniform() <= theta_1):
#                     a_1 = a_1+1
                    
#                     if (np.random.uniform() <= theta_head):
#                         a_2 = a_2+1
#                     else:
#                         b_2 = b_2+1
#                 else:
#                     b_1 = b_1+1
#             elif flip == 2:
#                 n_2_Bayesian_graph[count,i] = n_2_Bayesian_graph[count,i]+1
#                 if (np.random.uniform() <= theta_head):
#                     a_2 = a_2+1
#                 else:
#                     b_2 = b_2+1
        
#         error_1_Bayesian[count,i] = abs((a_1-1)/(a_1+b_1-2) - theta_1)
#         error_2_Bayesian[count,i] = abs((a_2-1)/(a_2+b_2-2) - theta_head)
        
# n_1_Bayesian_graph_average = np.average(n_1_Bayesian_graph,axis=1)
# n_2_Bayesian_graph_average = np.average(n_2_Bayesian_graph,axis=1)               

# print("--- %s seconds ---" % (time.time() - start_time))

#%% Empirical CI 2-node graph
        
# start_time = time.time()

# n_1_Empirical_graph = np.zeros((theta_2.shape[0],M))
# n_2_Empirical_graph = np.zeros((theta_2.shape[0],M))

# error_1_Empirical_graph = np.zeros((theta_2.shape[0],M))        
# error_2_Empirical_graph = np.zeros((theta_2.shape[0],M))


# for i in range(0,M):
    
#     for count, theta_head in enumerate(theta_2):
        
#         n1_history = list()
#         n2_history = list()       
                
#         for j in range(0,N):
            
#             t2 = theta_head*(var_emp(n2_history)-var_emp(n2_history+[1]))+(1-theta_head)*(var_emp(n2_history)-var_emp(n2_history+[0]))
#             t1 = theta_1*(var_emp(n1_history)-var_emp(n1_history+[1])+t2)+(1-theta_1)*(var_emp(n1_history)-var_emp(n1_history+[0]))
            
#             if t1 == t2:
#                 if(np.random.uniform() >=0.5):
#                     flip = 1
#                 else:
#                     flip = 2
#             elif t1 > t2:
#                 flip = 1
#             elif t1 < t2:
#                 flip = 2
                
#             if flip == 1:
#                 n_1_Empirical_graph[count,i] = n_1_Empirical_graph[count,i]+1
#                 if (np.random.uniform() <= theta_1):
#                     n1_history.append(1)
                    
#                     if (np.random.uniform() <= theta_head):
#                         n2_history.append(1)
#                     else:
#                         n2_history.append(0) 
#                 else:
#                     n1_history.append(0)
                
#             elif flip == 2:
#                 n_2_Empirical_graph[count,i] = n_2_Empirical_graph[count,i]+1
#                 if (np.random.uniform() <= theta_head):
#                     n2_history.append(1)
#                 else:
#                     n2_history.append(0)
            
#         # error_1_Empirical_graph[count,i] = abs(sum(n1_history)/n_1_Empirical_graph[count,i] - theta_1)
#         # error_2_Empirical_graph[count,i] = abs(sum(n2_history)/n_2_Empirical_graph[count,i] - theta_head)


# n_1_Empirical_graph_average = np.average(n_1_Empirical_graph, axis=1)
# n_2_Empirical_graph_average = np.average(n_2_Empirical_graph, axis=1)


# print("--- %s seconds ---" % (time.time() - start_time))        


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
    
e1 = np.maximum(np.quantile(error_1_Hoeffding,0.95,axis=1),np.quantile(error_2_Hoeffding,0.95,axis=1))

e2 = np.maximum(np.quantile(error_1_Bernstein,0.95,axis=1),np.quantile(error_2_Bernstein,0.95,axis=1))

e3 = np.maximum(np.quantile(error_1_Empirical,0.95,axis=1),np.quantile(error_2_Empirical,0.95,axis=1))

e4 = np.maximum(np.quantile(error_1_Bayesian,0.95,axis=1),np.quantile(error_2_Bayesian,0.95,axis=1))


e0 = np.maximum(error_1_naive,error_2_naive)

e5 = np.maximum(error_1_Hoeffding_theory,error_2_Hoeffding_theory)

e6 = np.maximum(error_1_Bernstein_theory,error_2_Bernstein_theory)

e7 = np.maximum(error_1_Empirical_theory,error_2_Empirical_theory)

e8 = np.maximum(error_1_Bayesian_theory,error_2_Bayesian_theory)

fig, ax = plt.subplots()

# ax.plot(theta_2, e1/e2, label = 'ratio')

# ax.plot(theta_2, e3/e4, label = 'ratio_theory')

# ax.plot(theta_2, e1, label = 'e_Hoeffding')

# ax.plot(theta_2, e2, label = 'e_Bernstein')

# ax.plot(theta_2, e3, label = 'e_Empirical')

# ax.plot(theta_2, e4, label = 'e_Bayesian')



ax.plot(theta_2, e5, label = 'e_Hoeffding Theory')

ax.plot(theta_2, e6, label = 'e_Bernstein Theory')

ax.plot(theta_2, e7, label = 'e_Empirical Theory')

ax.plot(theta_2, e8, label = 'e_Bayesian Theory')

ax.plot(theta_2, e0, label = 'e_naive Theory')

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
