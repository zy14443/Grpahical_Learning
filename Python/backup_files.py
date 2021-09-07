import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import math
import random
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_mlp(dim_list, activation='relu', batch_norm=False, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)# -*- coding: utf-8 -*-

#%% Model Failure Simulation
A = np.array([[0,1,1,1],
              [1,0,1,0],
              [1,1,0,0],
              [1,0,0,0]])

#A = np.array([[0,1,1],
#              [1,0,1],
#              [1,1,0]])

#A = np.array([[0,1],
#              [1,0]])

N = A.shape[0]
p0 = 0.3
p1 = 0.6

counter = 0
K=100
T=10
S_final = np.zeros((T,N,K))

S_0 = np.ones(N, dtype=int)
S_all_0 = np.array(S_0)

S_1 = np.array(S_0)
while(np.sum(S_1)==N):
    for i in range(0,N):
        t = np.random.uniform()
        if(t<p0):
            S_1[i]=0
            break
    S_all_0 = np.vstack((S_all_0,S_1))

while (counter < K):
    
    
    #print(S_all_0)
    
    S_t = np.array(S_1)
    S_all_1 = np.array(S_1)
    
    while(np.sum(S_t)>0):
    
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
        S_all_1 = np.vstack((S_all_1, S_t))
    
    
    if(S_all_1.shape[0]<T):
        S_all_1 = np.vstack((S_all_1, np.zeros((T-S_all_1.shape[0],N))))
    
#    print(S_all_1.shape)
    S_final[:,:,counter] = S_all_1
    counter = counter+1
    
#%%
# coint flip
N = 1000
batch_size = 10
S_c_all = np.zeros((batch_size,N))
p_h_all = np.zeros((batch_size,1))
for j in range(0,batch_size):
    p_h = np.random.uniform()
    S_c = np.zeros(N)
    for i in range(1,N):
        t = np.random.uniform()
        if(t<p_h):
            S_c[i]=1
    S_c_all[j,:] = np.sort(S_c)
    p_h_all[j,:] = p_h

input_tensor = torch.as_tensor(S_c_all, dtype = torch.float32).to(device)
target_tensor = torch.as_tensor(p_h_all, dtype = torch.float32).to(device)
#target_tensor = target_tensor.unsqueeze(0)

class Coin_Bias(nn.Module):
    def __init__ (self, device, N):
        super(Coin_Bias, self).__init__()       
        self.device = device    
        self.input_encoder = make_mlp([N,1], activation='sigmoid', batch_norm=False, dropout=0)
#        self.input_encoder = nn.Linear(N,1)
        
    def forward(self, input_tensor,batch_size):       
        output = self.input_encoder(input_tensor)
        return output  

coin_bias = Coin_Bias(device,N).to(device)
learning_rate=0.001 
optimizer = optim.Adam(coin_bias.parameters(), lr=learning_rate)

epoch_start = 0
n_epoch = 300
for epoch in range(epoch_start+1,n_epoch+1):
    optimizer.zero_grad()
    output = coin_bias(input_tensor, N)
    
    criterion = nn.MSELoss()
    loss = criterion(target_tensor, output)
    loss.backward()
    optimizer.step()
    
    print('train: (%d %d%%) %.4f' % (epoch, float(epoch-epoch_start) / (n_epoch-epoch_start) * 100, loss))
    
p_t = 0.58
S_t = np.zeros(N)
for i in range(1,N):
    t = np.random.uniform()
    if(t<p_t):
        S_t[i]=1

S_t = np.sort(S_t)
        
with torch.no_grad():
    test_tensor = torch.as_tensor(S_t, dtype = torch.float32).to(device)
    target_tensor = torch.as_tensor(p_t, dtype = torch.float32).to(device)
    target_tensor = target_tensor.unsqueeze(0)
    test_output = coin_bias(test_tensor, N)
    criterion = nn.MSELoss()
    loss = criterion(target_tensor, test_output)
    print(test_output)

#%%
# MLE
S_final.shape

p0_t = torch.as_tensor(0.5, dtype = torch.float32)
p1_t = torch.as_tensor(0.5, dtype = torch.float32)
p1_t.requires_grad=True
log_l = torch.zeros(1)
counter=0
while(counter<100):
    for i in range(0,S_final.shape[2]):
        
        S_i = S_final[:,:,i]
        t_fail = np.zeros(N, dtype=int)
        for j in range(0,N):
            t_fail[j] = np.where(S_i[:,j]==0)[0][0]
    #        print(t_fail)
        tf = max(t_fail)
        log_l = torch.log(torch.pow(p1_t,int(tf-1))*p1_t)+log_l
    log_l.backward()
    with torch.no_grad():
        p1_t = p1_t+learning_rate*p1_t.grad
    print(p1_t)
    p1_t.grad.data.zero_()
    counter = counter+1


#%%

your_win = 0
my_win = 0

counter = 0
history = np.zeros((window_size))

while counter < 300:
    
    user_input = input('Input 1 or 2: ')
    print('your input: '+user_input)
    
    if user_input == '1':
        t_input = 1
    elif user_input == '2':
        t_input = 2
    else: 
        continue
    
    (guess, history) = nn_predictor(t_input, history)
    
    
    print (guess, t_input)
    
    counter = counter+1


your_win = 0
my_win = 0
counter = 0

player_record = np.ones((8,2))*-1
wl_history = np.ones(2)*-1
input_history = np.ones(2)*-1


while counter < 100:
    user_input = input('Input 1 or 2: ')
    print('your input: '+user_input)
    
    if user_input == '1':
        t_input = 1
    elif user_input == '2':
        t_input = 2
    else: 
        continue
    
    (guess, player_record, wl_history, input_history) = mind_reader(t_input, player_record, wl_history, input_history)
    
    print (guess, t_input)
    
    counter = counter+1
    

