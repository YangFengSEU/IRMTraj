#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import os
import torch
import math
from unidataset_interaction import UniDataset
from matplotlib import pyplot as plt
from matplotlib import font_manager
'''
trSet=UniDataset('/home/lb/Documents/unidataset/lastversion/processed/zhidao/train/')
length=len(trSet)
rad=np.zeros([1])
psi_rad1=[]
for i in range(100):
    data=trSet[i]
    _, _, num, psi_rad, _=data
    for k in range(psi_rad.shape[0]):
        for index,j in enumerate(np.isnan(psi_rad[k])):
             if j==True:
                 continue
             psi_rad1.append(psi_rad[k][index])
        if psi_rad1==[]:
            continue
        psi_radtotal=np.stack(psi_rad1)
        
        rad=np.concatenate((rad,psi_radtotal),axis=0)
        psi_rad1=[]
'''
trSet=UniDataset('/home/lb/Documents/unidataset/lastversion/processed/miniinteraction/train/')
length=len(trSet)
rad=np.zeros([1])
psi_rad1=[]
for i in range(length):
    data=trSet[i]
    _, _, num, psi_rad, _,_=data

    for index,j in enumerate(np.isnan(psi_rad)):
         if j==True:
             continue
         psi_rad1.append(psi_rad[index])
    if psi_rad1==[]:
        continue
    psi_radtotal=np.stack(psi_rad1)
    
    rad=np.concatenate((rad,psi_radtotal),axis=0)
    psi_rad1=[]
d=0.05
num_bins=int((max(rad)-min(rad))//d)
plt.figure(figsize=(20,8),dpi=80)
plt.hist(rad,num_bins)

plt.xticks(range(-6,7,1))

plt.grid(alpha=0.4)
plt.show()
