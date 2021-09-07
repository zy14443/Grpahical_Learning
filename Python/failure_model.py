# -*- coding: utf-8 -*-

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
import matplotlib
matplotlib.axes.Axes.plot
matplotlib.pyplot.plot
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend


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
    return nn.Sequential(*layers)

#%%
   
def failure_model_1(N,T,K,p_conn,p_1):
    A = np.zeros((N,N), dtype=int)
    
    for i in range(0,N):
        for j in range(i+1,N):
            t = np.random.uniform()
            if t<p_conn:
                A[i,j] = 1
                A[j,i] = 1
       
    
    S_0 = np.ones(N, dtype=int)
#    S_0[np.random.choice(N,1)] = 0 #random initial failure
    S_0[3] = 0
    
    S_final = np.zeros((T,N,K))
    N_fail = np.zeros((T,K))
    
    n_sim=0
    while (n_sim < K):
        S_t = np.array(S_0)
        S_all_1 = np.array(S_0)
        n_fail_t = np.zeros((T))
        n_fail_t[0] = 1
        counter = 1
        while (counter < T):
            S_t_new = np.array(S_t)
            for i in range(0,N):
                if (S_t[i] == 1):
                    n_node = np.where(A[:,i]==1)
                    fail_node = np.where(S_t==0)
                    n_fail = np.intersect1d(n_node,fail_node).size
                    if (n_fail>0):
                        p_fail = 1-pow((1-p_1),n_fail)
                        t = np.random.uniform()
        #                print(i,p_fail,t)
                        if(t<p_fail):
                            S_t_new[i]=0    
            S_t = np.array(S_t_new)
            S_all_1 = np.vstack((S_all_1, S_t))
            n_fail_t[counter] = N-np.sum(S_t)
            counter = counter + 1
    
        S_final[:,:,n_sim] = S_all_1
        N_fail[:,n_sim] = n_fail_t
        n_sim = n_sim+1
    
    return (S_final, N_fail)

#%%
N = 5 # nubmer of nodes
T = 10 # number of time steps
K = 100 # number of simulations


p_conn = 0.5
#p_1 = 0.5

for p_1 in np.arange(0.1,0.9,0.1):
    (S_final, N_fail) = failure_model_1(N,T,K,p_conn,p_1)
    
    plt.plot(np.arange(0,T),np.average(N_fail,axis=1))
    plt.xlabel('Time Step')
    plt.ylabel('N failed nodes')


#p_conn = 0.5
p_1 = 0.5

for p_conn in np.arange(0.5,0.9,0.1):
    (S_final, N_fail) = failure_model_1(N,T,K,p_conn,p_1)
    
    plt.plot(np.arange(0,T),np.average(N_fail,axis=1))
    plt.xlabel('Time Step')
    plt.ylabel('N failed nodes')

#%%
INPUT_DIM = N
OUTPUT_DIM = N

ENC_EMB_DIM = 128
ENC_HID_DIM = 128

ENC_DROPOUT = 0

N_LAYER = 2
batch_size = K


class RNN_Model(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, RNN_type, device, num_layers=1, dropout=0.0):
        super(RNN_Model, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.num_layers = num_layers        
        self.RNN_type = RNN_type              

        self.input_encoder = make_mlp([input_dim,emb_dim], activation='relu', batch_norm=False, dropout=0)
        self.out_decoder = make_mlp([enc_hid_dim,input_dim], activation='relu', batch_norm=False, dropout=0)        

        self.gru = nn.GRU(emb_dim, enc_hid_dim, num_layers, dropout=dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = device
        

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.input_encoder(input)
            
        if self.RNN_type == 'GRU':
            output, hidden = self.gru(embedded, hidden)       
        
        output = output.squeeze(0)
        output = self.out_decoder(output)                
        return output, hidden

    def initHidden(self,batch_size):
        if self.RNN_type == 'GRU':
            return torch.zeros(self.num_layers, batch_size, self.enc_hid_dim, device=self.device)


class loss_cosine_MSE_seq():
    def __init__(self, beta):
        self.beta = beta

    def loss_fn(self, outputs, labels):        
#        labels = labels.permute(1,0,2) # [length, batch_size, ]
        target_len = labels.shape[0]
        batch_size = labels.shape[1]
        n_node = labels.shape[2]
        
#        outputs = outputs[0:target_len,:] # [length, batch_size, ]
        
        loss_total = 0
        for i in range(target_len):
            cos = nn.CosineSimilarity(dim=1, eps=1e-8)
            loss_1 = (1-cos(outputs[i], labels[i]))/2
 
            mse = nn.PairwiseDistance()
            loss_2 = mse(outputs[i], labels[i])
            loss_2 = (loss_2**2)/n_node
        
            loss_total += sum(self.beta*loss_1 + (1-self.beta)*loss_2)/batch_size
    
        return loss_total/target_len

class Seq_Generator(nn.Module):
    def __init__ (self, encoder, device):
        super(Seq_Generator, self).__init__()
        
        self.encoder = encoder
        self.device = device    
    
    def forward(self, input_tensor, T, batch_size, teacher_forcing_ratio=0):
        
        outputs = torch.zeros(T-1, batch_size, N).to(device)
        
        encoder_hidden = enc.initHidden(batch_size)
        output = input_tensor[0]
        for t in range(0, T-1):
            output, encoder_hidden = enc(output, encoder_hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio     
            output = (input_tensor[t] if teacher_force else output)
        

        return outputs       


#%%

p_conn = 0.7
p_1 = 0.6
(S_final, N_fail) = failure_model_1(N,T,K,p_conn,p_1)

        
enc = RNN_Model(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, 'GRU', device, N_LAYER, ENC_DROPOUT).to(device)
Seq_Gen = Seq_Generator(enc,device).to(device)
learning_rate=0.0001 
optimizer = optim.Adam(Seq_Gen.parameters(), lr=learning_rate)
#teacher_forcing_ratio=0

input_tensor = torch.as_tensor(S_final, dtype = torch.float32).to(device)
input_tensor = input_tensor.permute(0,2,1)

epoch_start = 0
n_epoch = 300
for epoch in range(epoch_start+1,n_epoch+1):
    optimizer.zero_grad()
    
    
    outputs = Seq_Gen(input_tensor, T, batch_size)
    
    criterion = loss_cosine_MSE_seq(beta=0)
    loss = criterion.loss_fn(outputs, input_tensor[1:T,:,:])
    
    loss.backward()
    optimizer.step()
    
    print('train: (%d %d%%) %.4f' % (epoch, float(epoch-epoch_start) / (n_epoch-epoch_start) * 100, loss))

#%%
with torch.no_grad():
    outputs = outputs.permute(0,2,1)
    outputs_np = outputs.detach().cpu().numpy()

N_fail_pred = np.zeros(T)
N_fail_pred[0] = 1
N_fail_pred[1:T] = np.ones(T-1)*N - np.sum(outputs_np[:,:,0],axis=1)

fig, ax = plt.subplots()
ax.plot(np.arange(0,T),np.average(N_fail,axis=1), label = 'Ground Truth Average')
ax.plot(np.arange(0,T),(N_fail_pred), '--', label = 'Predicted Average')
legend = ax.legend(loc='lower right', shadow=False, fontsize='x-large')

plt.xlabel('Time Step')
plt.ylabel('N failed nodes')
    


#%%
#mind reading

window_size = 5
threshold = 0.5

#target_tensor = target_tensor.unsqueeze(0)

class Predict_Input(nn.Module):
    def __init__ (self, device, N):
        super(Predict_Input, self).__init__()       
        self.device = device    
        self.input_encoder = make_mlp([N,256,2], activation='tanh', batch_norm=False, dropout=0)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input_tensor,batch_size):       
        output = self.input_encoder(input_tensor)
        output = self.softmax(output)
        return output  

predictor = Predict_Input(device,window_size).to(device)
learning_rate=0.001 
optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

torch.save({'model_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, './initial_state.pth')

def nn_predictor (input, history, previous_loss, smooth_counter):
#    user_input = input('Input 1 or 2: ')
#    print('your input: '+user_input)
#    
#    if user_input != '1' and user_input != '2':
#        continue
    
    if input == 1:
        user_input = '1'
    elif input == 2:
        user_input = '2'
    
    
    p_s = 0.5
    t = np.random.uniform()
    if(t<p_s):
        guess = 1
    else:
        guess = 2   
      
    target_tensor = torch.as_tensor(int(user_input)-1, dtype = torch.long).to(device)
    target_tensor = target_tensor.unsqueeze(0)
    
    if counter>window_size:
        with torch.no_grad():
            input_tensor = torch.as_tensor(history, dtype = torch.float32).to(device)
            output = predictor(input_tensor, window_size)
            if (output[0] > output[1]):
                guess = 1
            else:
                guess = 2    
        
        epoch_start = 0
        n_epoch = 50
        for epoch in range(epoch_start+1,n_epoch+1):
            input_tensor = torch.as_tensor(history, dtype = torch.float32).to(device)
            optimizer.zero_grad()
            output = predictor(input_tensor, window_size)
            output = output.unsqueeze(0)
        
#        criterion = nn.MSELoss()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target_tensor)
                
                    
            loss.backward()
            optimizer.step()
        print (loss-previous_loss)
        if (loss-previous_loss) > threshold:
            smooth_counter = smooth_counter + 1
            if (smooth_counter == 10):
#                checkpoint = torch.load('./initial_state.pth')
#                predictor.load_state_dict(checkpoint['model_state_dict'])
#                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                smooth_counter = 0
            
        previous_loss = loss
        print (smooth_counter)
                
    
#    print('my guess: '+str(guess))
    
    if user_input == '1':
        if counter < window_size:
            history[counter] = 1
        else:
            history = np.roll(history,-1)
            history[window_size-1] = 1
#        if guess == 1:
#            my_win = my_win+1
#        elif guess == 2:
#            your_win = your_win+1
    elif user_input == '2':
        if counter < window_size:
            history[counter] = 2
        else:
            history = np.roll(history,-1)
            history[window_size-1] = 2
#        if guess == 2:
#            my_win = my_win+1
#        elif guess == 1:
#            your_win = your_win+1
    
#    print('you win '+str(your_win)+' times, lose '+str(my_win)+' times.' )
#    print(history)

    return (guess, history, previous_loss, smooth_counter)
 
#%%
# mind reading machine
def mind_reader(input, player_record, wl_history, input_history):   
    if input == 1:
        user_input = '1'
    elif input == 2:
        user_input = '2'
#    else:
#        guess = -1
#        return guess

    p_s = 0.5
    t = np.random.uniform()
    if(t<p_s):
        guess = 1
    else:
        guess = 2
    
    
    if input_history[0] == input_history[1]:
        if wl_history[0]==1 and wl_history[1]==1:
            if player_record[0][0] == player_record[0][1]:
                if player_record[0][1] == 1:
                    guess = input_history[1]
                elif player_record[0][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
                            
        elif wl_history[0]==1 and wl_history[1]==0:
            if player_record[1][0] == player_record[1][1]:
                if player_record[1][1] == 1:
                    guess = input_history[1]
                elif player_record[1][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
        elif wl_history[0]==0 and wl_history[1]==1:
            if player_record[2][0] == player_record[2][1]:
                if player_record[2][1] == 1:
                    guess = input_history[1]
                elif player_record[2][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
        elif wl_history[0]==0 and wl_history[1]==0:
           if player_record[3][0] == player_record[3][1]:
                if player_record[3][1] == 1:
                    guess = input_history[1]
                elif player_record[3][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
    else:
        if wl_history[0]==1 and wl_history[1]==1:
            if player_record[4][0] == player_record[4][1]:
                if player_record[4][1] == 1:
                    guess = input_history[1]
                elif player_record[4][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
        elif wl_history[0]==1 and wl_history[1]==0:
          if player_record[5][0] == player_record[5][1]:
                if player_record[5][1] == 1:
                    guess = input_history[1]
                elif player_record[5][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
        elif wl_history[0]==0 and wl_history[1]==1:
            if player_record[6][0] == player_record[6][1]:
                if player_record[6][1] == 1:
                    guess = input_history[1]
                elif player_record[6][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
        elif wl_history[0]==0 and wl_history[1]==0:
            if player_record[7][0] == player_record[7][1]:
                if player_record[7][1] == 1:
                    guess = input_history[1]
                elif player_record[7][1] == 0:
                    if input_history[1] == 2:
                        guess = 1
                    else:
                        guess = 2
    
    
    
#    print('my guess: '+str(guess))
    
    if user_input == '1' and input_history[1] == 1:
        sd_indicator = 1
    elif user_input == '2' and input_history[1] == 2:
        sd_indicator = 1
    else:
        sd_indicator = 0   
    
#    print(input_history)
#    print(wl_history)
    
    if input_history[0] == input_history[1]:
        if wl_history[0]==1 and wl_history[1]==1:
            player_record[0] = np.roll(player_record[0],-1)
            player_record[0][1] = sd_indicator#    if user_input != '1' and user_input != '2':
#        continue
    
        elif wl_history[0]==1 and wl_history[1]==0:
            player_record[1] = np.roll(player_record[1],-1)
            player_record[1][1] = sd_indicator
        elif wl_history[0]==0 and wl_history[1]==1:
            player_record[2] = np.roll(player_record[2],-1)
            player_record[2][1] = sd_indicator
        elif wl_history[0]==0 and wl_history[1]==0:
            player_record[3] = np.roll(player_record[3],-1)
            player_record[3][1] = sd_indicator
    else:
        if wl_history[0]==1 and wl_history[1]==1:
            player_record[4] = np.roll(player_record[4],-1)#    if user_input != '1' and user_input != '2':
#        continue
    
            player_record[4][1] = sd_indicator
        elif wl_history[0]==1 and wl_history[1]==0:
            player_record[5] = np.roll(player_record[5],-1)
            player_record[5][1] = sd_indicator
        elif wl_history[0]==0 and wl_history[1]==1:
            player_record[6] = np.roll(player_record[6],-1)
            player_record[6][1] = sd_indicator
        elif wl_history[0]==0 and wl_history[1]==0:
            player_record[7] = np.roll(player_record[7],-1)
            player_record[7][1] = sd_indicator
    
    
#    print(player_record)
    
    
    if user_input == '1':
        
        input_history = np.roll(input_history,-1)
        input_history[1] = 1
        
        if guess == 1:
#            my_win = my_win+1
            wl_history = np.roll(wl_history,-1)
            wl_history[1] = 1
        elif guess == 2:
#            your_win = your_win+1
            wl_history = np.roll(wl_history,-1)
            wl_history[1] = 0
    elif user_input == '2':
        
        input_history = np.roll(input_history,-1)
        input_history[1] = 2
        
        if guess == 2:
#            my_win = my_win+1
            wl_history = np.roll(wl_history,-1)
            wl_history[1] = 1
        elif guess == 1:
#            your_win = your_win+1
            wl_history = np.roll(wl_history,-1)
            wl_history[1] = 0 
           
#    print('you win '+str(your_win)+' times, lose '+str(my_win)+' times.' )

    return (guess, player_record, wl_history, input_history)


#%%
    
win_1 = 0
win_2 = 0

counter = 0
history = np.zeros((window_size))
previous_loss = 999
smooth_counter = 0

player_record = np.ones((8,2))*-1
wl_history = np.ones(2)*-1
input_history = np.ones(2)*-1


seq_len = 300
gt_sequence = np.tile([2,1,1,2,2],seq_len)


while counter < seq_len:
    
    t_input = gt_sequence[counter]
#    user_input = input('Input 1 or 2: ')
#    print('your input: '+user_input)
#    
#    if user_input == '1':
#        t_input = 1
#    elif user_input == '2':
#        t_input = 2
#    else: 
#        continue
    
    (guess_1, history,previous_loss,smooth_counter) = nn_predictor(t_input, history,previous_loss,smooth_counter)
    
    (guess_2, player_record, wl_history, input_history) = mind_reader(t_input, player_record, wl_history, input_history)
   
    print (guess_1, guess_2, t_input)
    
    if(guess_1 == t_input):
        win_1 = win_1 +1
    if(guess_2 == t_input):
        win_2 = win_2 +1
    
#    print(win_1,win_2)
    
    counter = counter+1

print(win_1/seq_len,win_2/seq_len)

#%%

your_win = 0
my_win = 0

counter = 0
history = np.zeros((window_size))
previous_loss = 999
while counter < 300:
    
    user_input = input('Input 1 or 2: ')
    print('your input: '+user_input)
    
    if user_input == '1':
        t_input = 1
    elif user_input == '2':
        t_input = 2
    else: 
        continue
    
    (guess, history, previous_loss) = nn_predictor(t_input, history, previous_loss)
    
    
    print (guess, t_input)
    
    counter = counter+1