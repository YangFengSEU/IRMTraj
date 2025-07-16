#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import numpy as np
import os
import torch
import math
import sys

from matplotlib import pyplot as plt

from map_api import InterMap
from utils.mpl_plotting_utils import visualize_centerline
from torch.utils.data import DataLoader
import pickle as pkl

class UniDataset1:
    def __init__(self, data_path, hist_len = 10, fut_len = 15 , enc_size=64,batch_size = 20):
        self.dir = data_path
        self.data_list = []
        self.leng = 100000
        
        self.list = os.listdir(self.dir)
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.total_len = hist_len + fut_len
        
        self.enc_size=enc_size
        
        
        
    def __len__(self):
        return len(self.list)
    
    
    def __getitem__(self, idx):
        with open(os.path.join(self.dir,self.list[idx]),'rb') as f:
            data = pkl.load(f)

        hist, fut, num, nbrs, vehtype, op_mask, mapfeats, motiva = data
        return hist, fut, num, nbrs, vehtype, op_mask, mapfeats, motiva
        

    def collate_fn(self, samples):
        veh_num = 0
        samplenum = 0
        for _, _, num,nbrs, _, _, _, _ in samples:
            for j in nbrs: 
                veh_num += sum([len(j[i])!=0 for i in range(len(j))])

            samplenum += num
        nbrs_batch = torch.zeros(self.hist_len, veh_num, 2)
        
        # Initialize 
        hist_batch = torch.zeros(self.hist_len, samplenum, 2)
        fut_batch = torch.zeros(self.fut_len, samplenum, 2)
        op_mask_batch = torch.ones(self.fut_len, samplenum, 2)
        
        pos=[0,0]
        mask_batch = torch.zeros(samplenum, 13, 13, self.enc_size)
        mask_batch = mask_batch.byte()
        
        mapfeats_batch=torch.zeros(samplenum,5,10,2)
        motiva_batch=torch.zeros(samplenum,6)
        
        count = 0
        sample_count = 0
        for hist, fut, num, nbrs,_ , op_mask, mapfeats_list, motiva in samples:
            
            for k in range(num):

                hist_batch[0:10, sample_count + k, :] = torch.from_numpy(hist[k].astype(float))
                fut_batch[0:15, sample_count + k, :] = torch.from_numpy(fut[k].astype(float))
                
                # print(op_mask.shape)
                op_mask_batch[0:15, sample_count + k,:] = torch.from_numpy(op_mask[k].astype(int))
                motiva_batch[sample_count + k,:] = torch.from_numpy(motiva[k].astype(float))
                m=mapfeats_list[k].shape[0]
                if m>=5:
                    mapfeats_batch[sample_count + k,:,:,:]=torch.from_numpy(mapfeats_list[k][:5,:,:].astype(float))
                else:
                    mapfeats_batch[sample_count + k,0:m,:,:]=torch.from_numpy(mapfeats_list[k][:,:,:].astype(float))
                   
            # Set up neighbor, neighbor sequence length, and mask batches:
            for k, singlenbr in enumerate(nbrs):
                for id,nbr in enumerate(singlenbr):
                    if len(nbr)!=0:
                        nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0].astype(float))
                        nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1].astype(float))
                        pos[0] = id % 13
                        #print(pos[0])
                        pos[1] = id // 13
                        #print(pos[1])
                        mask_batch[sample_count + k,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                        count+=1
            sample_count += num
            

            
        return hist_batch, nbrs_batch, mask_batch, fut_batch, op_mask_batch, mapfeats_batch, motiva_batch
        

        
                    
   
