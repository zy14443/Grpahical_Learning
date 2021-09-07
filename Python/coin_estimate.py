#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:37:55 2019

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

p_fail = 0.5

N = 25

K = 1000
S_mean = np.zeros((K,1))

for j in range(0,K):
    S_t = np.zeros((N,1),dtype=int)
    for i in range(0,N):
        t = np.random.uniform()
        if(t<p_fail):
            S_t[i]=1

    S_mean[j] = np.mean(S_t)

np.std(S_mean)

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

n, bins, patches = plt.hist(x=S_mean,bins=[0,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.96,1.05])

#%%

import numpy as np
import matplotlib.pyplot as plt


N = 10 # Number of flips
BIAS_HEADS = 0.3 # The bias of the coin


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
    plt.pause(0.05)
    

plt.show()

#%%