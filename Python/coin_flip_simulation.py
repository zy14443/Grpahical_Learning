import numpy as np
import matplotlib.pyplot as plt


N = 200 # Number of flips
BIAS_HEADS = 0.1 # The bias of the coin


bias_range = np.linspace(0, 1, 101) # The range of possible biases
prior_bias_heads = np.ones(len(bias_range)) / len(bias_range) # Uniform prior distribution
flip_series = (np.random.rand(N) <= BIAS_HEADS).astype(int) # A series of N 0's and 1's (coin flips)

for flip in flip_series:
    likelihood = bias_range**flip * (1-bias_range)**(1-flip)
    evidence = np.sum(likelihood * prior_bias_heads)
    prior_bias_heads = likelihood * prior_bias_heads / evidence

plt.plot(bias_range, prior_bias_heads)
plt.xlabel('Heads Bias')
plt.ylabel('P(Heads Bias)')
plt.grid()
plt.show()

#%%
# probability bayesian approach
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

M = 800 # number of experiments
N = 800 # number of flips per experiment

h = np.ones((N,M))

h_2 = np.ones((N,M))

bias_range = np.linspace(0.1,0.9,50)
h_average = np.ones((bias_range.shape[0],N))

h_average_2 = np.ones((bias_range.shape[0],N))

bias_count = 0
for theta_head in bias_range:
    for i in range(0,M):
        flip_series = (np.random.rand(N) <= theta_head).astype(int) # A series of N 0's and 1's (coin flips)
        
        a = 1
        b = 1
        
        counter = 0
        for flip in flip_series:
            h_2[counter,i] = (beta.cdf(theta_head+0.05,a,b) - beta.cdf(theta_head-0.05,a,b))
            
        
            h[counter,i] = beta.var(a,b) 
            # h[counter] = beta.entropy(a,b) - theta_head*beta.entropy(a+1,b) - (1-theta_head)*beta.entropy(a,b+1)
            counter = counter+1
            if flip == 1:
                a = a+1
            elif flip == 0:
                b = b+1
                
    # plt.plot(np.linspace(1, N, N), np.average(h,axis=1)) 
    h_average[bias_count,:] = np.average(h,axis=1)
    h_average_2[bias_count,:] = np.average(h_2,axis=1)

    
    
    bias_count = bias_count+1

#%%
threshold_1 = np.ones((bias_range.shape[0],1))
for i in range(0,bias_range.shape[0]):
    for j in range(0,N):
        if(h_average[i,j]<0.00035):
            threshold_1[i] = j
            break
plt.plot(bias_range, threshold_1)         


threshold_3 = np.ones((bias_range.shape[0],1))
for i in range(0,bias_range.shape[0]):
    for j in range(0,N):
        if(h_average_2[i,j]>0.94):
            threshold_3[i] = j
            break
plt.plot(bias_range, threshold_3)

# entropy = np.ones((bias_range.shape[0],1))
# for i in range(0,bias_range.shape[0]):
#     entropy[i] =  beta.entropy(bias_range[i]*1000,(1-bias_range[i])*1000)
        
        
# plt.plot(bias_range, entropy)           

#%%
# probability frequentist approach
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

#Bernstein Inequality
bias_range = np.linspace(0.1,0.9,50)
threshold = np.ones((bias_range.shape[0],1))

epsilon = 0.05
delta = 1 - 0.95

threshold_H = np.ones((bias_range.shape[0],1))
threshold_B = np.ones((bias_range.shape[0],1))
for i in range(0,bias_range.shape[0]):
    theta_head = bias_range[i]
    threshold_H[i] = np.log(2/delta) / (2*epsilon*epsilon)
    threshold_B[i] = np.log(2/delta)*(2*theta_head*(1-theta_head)+2*epsilon/3)/(epsilon*epsilon) #Bernstein Inequality
    
fig, ax = plt.subplots()
ax.plot(bias_range, threshold_H, label = 'Hoeffding')
ax.plot(bias_range, threshold_B, label = 'Bernstein')
ax.plot(bias_range, threshold_3, label = 'Bayesian')
legend = ax.legend(loc='upper right', shadow=False, fontsize='x-small')

plt.xlabel('Bias_head')
plt.ylabel('N_k')


#%%
#mutual information
from scipy.stats import beta
import matplotlib.pyplot as plt

M=200
N = 100

for theta_head in [0.1,0.5]:
    h = np.ones((N,M))
    for i in range(0,M):
        flip_series = (np.random.rand(N) <= theta_head).astype(int) # A series of N 0's and 1's (coin flips)

        a = 1
        b = 1
        
        counter = 0
        for flip in flip_series:
            # h[counter,i] = 1- (beta.cdf(theta_head+0.05,a,b) - beta.cdf(theta_head-0.05,a,b))
            
            h[counter,i] = beta.entropy(a,b)
            if flip == 1:
                # h[counter,i] = beta.entropy(a,b) - beta.entropy(a+1,b)
                a = a+1
            elif flip == 0:
                # h[counter,i] = beta.entropy(a,b) - beta.entropy(a,b+1)
                b = b+1
            counter = counter+1
                
    plt.plot(np.linspace(1, N, N), np.average(h,axis=1)) 
    
    # plt.plot(np.linspace(2, N, N-1), -np.diff(np.average(h,axis=1)) ) 

    
#%%
a = 500
b = 500
x = np.linspace(beta.ppf(0.01, a, b),
               beta.ppf(0.99, a, b), 100)
fig, ax = plt.subplots(1, 1)
ax.plot(x, beta.pdf(x, a, b),
         'r-', lw=5, alpha=0.6, label='beta pdf')