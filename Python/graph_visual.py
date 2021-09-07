#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:52:12 2019

@author: zheng.1443
"""

# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

pd.options.display.max_columns = 20

df = pd.DataFrame([[0,1,1,0], [0,0,0,1], [0,0,0,1],[0,0,0,0]]) # diamond

df = pd.DataFrame([[0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]]) # Loop

df = pd.DataFrame([[0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0]]) # Chain

df = pd.DataFrame([[0,1,1,0,0,0,0], [0,0,0,1,1,0,0], [0,0,0,0,0,1,1],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]) #tree

df = pd.DataFrame([[0,1,1,1,1,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]) #center

#%%
G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())

print(nx.info(G))

 
# And a data frame with characteristics for your nodes
carac = pd.DataFrame({ 'ID':[0, 1 ,2, 3, 4,5,6,7], 'myvalue':[0,1,1,1,0,1,1,1] })
 
G.nodes()
G.edges()
# Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!
 
# Here is the tricky part: I need to reorder carac to assign the good color to each node
carac= carac.set_index('ID')
carac=carac.reindex(G.nodes())
 
# And I need to transform my categorical column in a numerical value: group1->1, group2->2...
carac['myvalue']=pd.Categorical(carac['myvalue'])
carac['myvalue'].cat.codes
 
# Custom the nodes:
nx.draw(G, with_labels=True, pos = nx.planar_layout(G), node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=1500)

#nx.draw(G, with_labels=True, pos = nx.spring_layout(G), node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=1500)