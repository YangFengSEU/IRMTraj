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

class UniDataset:
    def __init__(self, folder1,folder2, hist_len = 10, fut_len = 15 , enc_size=64,city1="FT",city2="EP",obs_range=30):
        self.dir1 = folder1
        self.dir2 = folder2
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.total_len = hist_len + fut_len
        
        self.enc_size=enc_size
        
        self.data_list1 = os.listdir(self.dir1)
        self.data_list2 = os.listdir(self.dir2)
        
        self.map=InterMap()
        
        self.city1=city1
        self.city2=city2
        
        self.obs_range = obs_range
        print(len(self.data_list1))
        print(len(self.data_list2))
        
    def __len__(self):

        if len(self.data_list1)>len(self.data_list2):
            l=len(self.data_list2)
        else:
            l=len(self.data_list1)
        
        return l 
    
    
    def extend_node_of_line(self,line,feature,roadindex,num):
        reline=line
        flag=1
        
        while reline.shape[0]<num:
            a=np.zeros([reline.shape[0]-1])
            for i in range(reline.shape[0]-1):
                a[i]=(reline[i,0]-reline[i+1,0])*(reline[i,0]-reline[i+1,0])+(reline[i,1]-reline[i+1,1])*(reline[i,1]-reline[i+1,1])
            maxid=np.argmax(a)
            while flag==1:
                if roadindex[maxid]==roadindex[maxid+1]:
                    flag=0
                else:
                    a[maxid]=0;
                    maxid=np.argmax(a)
                    
            x=(reline[maxid,0]+reline[maxid+1,0])/2
            y=(reline[maxid,1]+reline[maxid+1,1])/2
            reline = np.insert(reline,(maxid+1)*2,[x,y]).reshape(-1,2)
            roadindex=np.insert(roadindex,maxid+1,roadindex[maxid])
            x=feature[maxid,0]/2
            y=feature[maxid,1]/2
            feature[maxid,0]=x
            feature[maxid,1]=y
            feature = np.insert(feature,(maxid+1)*2,[x,y]).reshape(-1,2)
            
        return reline,feature,roadindex
    def addnode_from_ctrln(self,ctr,num):
        
        while ctr.shape[0]<num:
            n=ctr.shape[0]
            for i in range(0,n-1):
                if ctr.shape[0]<num:
                    x=(ctr[2*i+1,0]+ctr[2*i,0])/2
                    y=(ctr[2*i+1,1]+ctr[2*i,1])/2
                    ctr=np.insert(ctr,(2*i+1)*2,[x,y]).reshape(-1,2)
                else:
                    break
        return ctr
    
    def delnode_from_ctrln(self,ctr,num):
        
        while ctr.shape[0]>num:
            n=ctr.shape[0]
            for i in range(0,int(n/2)-1):
                if ctr.shape[0]>num:
                   ctr=np.delete(ctr,i+1,0)
                else:
                    break
        return ctr
    
    def item_from_dir(self,idx,datasetid,icity):
        if datasetid==1:
            directoin=self.dir1
            ori = np.load(directoin + self.data_list1[idx], allow_pickle = True)
        else:
            directoin=self.dir2
            ori = np.load(directoin + self.data_list2[idx], allow_pickle = True)
        
        
        hist = ori[:, :self.hist_len, 3:5]#3s-5s
        
        op_mask = ori[:, self.hist_len:, -1:]
        fut = ori[:, self.hist_len:, 3:5] - ori[:, self.hist_len-1:self.hist_len, 3:5]
        
        vehtype=ori[:,0,4]
    
        num = hist.shape[0]
        
        dest=fut[:,-1,:]
        
        #res=0
        dest_mask = []
        for x in range(num):
            if not math.isnan(dest[x][0]):
                dest_mask.append(x)
        R_list = []
        T_list = []
        for i in range(num):
            norm_v = hist[i, -1, :]-hist[i, 0, :]
            R_i =np.zeros([2,2])
            l = math.sqrt(norm_v[0]*norm_v[0]+norm_v[1]*norm_v[1])
            R_i[0][0] = norm_v[1]/l
            R_i[1][0] = -norm_v[0]/l
            R_i[0][1] = norm_v[0]/l
            R_i[1][1] = norm_v[1]/l
            T_i =  -hist[i, -1, :]
            R_list.append(R_i)
            T_list.append(T_i)                        
        #get grid
        grid = np.zeros([num, 13, 13])-1
        for i in range(num):
                x_i = hist[i, -1, 0]
                y_i = hist[i, -1, 1]
                
                for j in range(num):
                    if j==i:
                        continue
                    
                    x_j = hist[j, -1, 0]
                    y_j = hist[j, -1, 1]
                
                    g_x = (x_j - x_i) / 5
                    g_y = (y_j - y_i) / 5
                    
                    if g_x > 0 and g_x < (13 - 1)/2:
                        g_x = math.ceil(g_x) + (13 - 1)/2
                    elif g_x < 0 and abs(g_x) < (13 - 1)/2:
                        g_x = math.floor(g_x) + (13 - 1)/2
                    else:
                        continue
                    
                    if g_y > 0 and g_y < (13 - 1)/2:
                        g_y = math.ceil(g_y) + (13 - 1)/2
                    elif g_y < 0 and abs(g_y) < (13 - 1)/2:
                        g_y = math.floor(g_y) + (13 - 1)/2
                    else:
                        continue
                    
                    # print(g_x, g_y)
                    grid[i, int(g_x), int(g_y)] = j
            
        nbrs = []
        motiva_list = []
        for i in range(num):
            grid_line = grid.reshape([num, -1])
            motiva = np.zeros([6,1])
            fut_norm = fut[int(i)]*R_list[i]+T_list[i]

            if fut_norm[-1,0]-fut_norm[0,0]>1:
                motiva[2] = 1
            elif fut_norm[-1,0]-fut_norm[0,0]<-1:
                motiva[0] = 1
            else:
                motiva[1] = 1
            v = fut_norm[1:,:]-fut_norm[:-1,:]
            if v[-1,1]-v[0,1]>0.1:
                motiva[3] = 1
            elif v[-1,1]-v[0,1]<-0.1:
                motiva[5] = 1
            else:
                motiva[4] = 1
            motiva_list.append(motiva)

            nbr = []
            for j in grid_line[i]:
                if not j == -1:
                   # print(hist[int(i)].shape)
                   traj = hist[int(j)]*R_list[i]+T_list[i]#
                   nbr.append(traj)
                else:
                    nbr.append(np.empty([0,2]))
            #nbr = np.stack(nbr)
            nbrs.append(nbr)
        


        mapfeats_list=[]

        tar_candts_list=[]

        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        #print(num)
        for i in range(num):
    
            end = hist[i,-1].copy().astype(np.float32)
            lane_ids = self.map.get_lane_ids_in_xy_bbox(end[0], end[1], icity, radius * 1)
            lane_ids = copy.deepcopy(lane_ids)
            lanes = dict()
            lane_distances=[]
            for lane_id in lane_ids:
                
                lane = self.map.city_lane_centerlines_dict[icity][lane_id]
                lane = copy.deepcopy(lane)
                
                #lane.centerline=lane.centerline - end.reshape(-1, 2)
                lane.centerline=lane.centerline*R_list[i] + T_list[i]
               
                lanes[lane_id] = lane
                lane_distance=(lane.centerline[0,0]*lane.centerline[0,0]+lane.centerline[0,1]*lane.centerline[0,1]+lane.centerline[-1,0]*lane.centerline[-1,0]+lane.centerline[-1,1]*lane.centerline[-1,1])/2
                lane_distances.append([lane_distance,lane_id])

            lane_distances=sorted(lane_distances,key=lambda x:x[0],reverse=False)#按距离顺序
            
            mapfeats=[]

            for _,lane_id in lane_distances:
                #print(lane_id)
                lane = lanes[lane_id]
                ctrln = lane.centerline
                if ctrln.shape[0]>11:
                    ctrln=self.delnode_from_ctrln(ctrln,11)
                elif ctrln.shape[0]<11:
                    ctrln=self.addnode_from_ctrln(ctrln,11)

                ctr=np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)
                #feat=np.asarray(ctrln[1:] - ctrln[:-1], np.float32)
                #mapfeat=np.concatenate((ctr, feat),axis=1)
                mapfeats.append(ctr)

            '''
            lane_idcs = []
            count = 0
            for j, ctr in enumerate(ctrs):
                lane_idcs.append(j * np.ones(len(ctr), np.int64))
                count += len(ctr)
            '''
            #lane_idcs = np.concatenate(lane_idcs, 0)        
            #ctrs=np.concatenate(ctrs, 0)
            #feats=np.concatenate(feats, 0)
            #ctrs,feats,lane_idcs=self.extend_node_of_line(ctrs,feats,lane_idcs,313)
            
            #lane_idcs_list.append(lane_idcs)
            mapfeats=np.array(mapfeats)
            mapfeats_list.append(mapfeats)
            
          
        hist=hist-ori[:, self.hist_len-1:self.hist_len, 3:5]
        
          
        return hist, fut, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask, motiva_list
    
    
    def __getitem__(self, idx):
        hist1, fut1, num1, nbrs1, vehtype1, op_mask1, mapfeats_list1, dest_mask1, motiva_list1=self.item_from_dir(idx, 1,self.city1)
        hist2, fut2, num2, nbrs2, vehtype2, op_mask2, mapfeats_list2, dest_mask2, motiva_list2=self.item_from_dir(idx,2,self.city2)
        traj1=[hist1, num1, nbrs1, op_mask1, mapfeats_list1]
        traj2=[hist2, num2, nbrs2, op_mask2, mapfeats_list2]

        env1={'traj':traj1,'fut':fut1,'deskmask':dest_mask1,'motiv':motiva_list1}
        env2={'traj':traj2,'fut':fut2,'deskmask':dest_mask2,'motiv':motiva_list2}
        #print(len([env1,env2]),'getitem')
        return [env1,env2]
        
    def collate_samples(self, env):
        veh_num = 0
        samplenum = 0
        maxmapnum=337
        hist, num, nbrs, op_mask, mapfeats_list=env['traj']

        traj=[[hist, num, nbrs, op_mask, mapfeats_list, env['fut'], env['deskmask'],env['motiv']]]
        #print(len(traj))
        #print('traj',len(traj[0][0]))

        for hist,num,nbrs,op_mask, _, _, _,_ in traj:
            for j in nbrs:
                veh_num += sum([len(j[i])!=0 for i in range(len(j))])
            samplenum += num
        nbrs_batch = torch.zeros(self.hist_len, veh_num, 2)

        # Initialize 
        hist_batch = torch.zeros(self.hist_len, samplenum, 2)
        #print(hist_batch)
        fut_batch = torch.zeros(self.fut_len, samplenum, 2)
        op_mask_batch = torch.ones(self.fut_len, samplenum, 2)
        op_mask_batch = torch.ones(self.fut_len, samplenum, 2)
        destmask_batch=[]
        
        pos=[0,0]
        mask_batch = torch.zeros(samplenum, 13, 13, self.enc_size)
        mask_batch = mask_batch.byte()
        
        mapfeats_batch=torch.zeros(samplenum,5,10,2)
        motiva_batch=torch.zeros(samplenum,6)

        count = 0
        sample_count = 0
        dest_chan=0
        for hist, num, nbrs, op_mask, mapfeats_list,  fut, dest_mask, motiva in traj:
            
            for k in range(num):

                hist_batch[0:15, sample_count + k, :] = torch.from_numpy(hist[k].astype(float))
                # print(op_mask.shape)
                op_mask_batch[0:25, sample_count + k,:] = torch.from_numpy(op_mask[k].astype(int))
                fut_batch[0:25, sample_count + k, :] = torch.from_numpy(fut[k].astype(float))
                motiva_batch[sample_count + k,:] = torch.from_numpy(motiva[k].astype(float))
                m=mapfeats_list[k].shape[0]
                if m>=5:
                    mapfeats_batch[sample_count + k,:,:,:]=torch.from_numpy(mapfeats_list[k][:5,:,:].astype(float))
                else:
                    mapfeats_batch[sample_count + k,0:m,:,:]=torch.from_numpy(mapfeats_list[k][:,:,:].astype(float))            
            for dest in dest_mask:
                dest_chan=dest+sample_count
                destmask_batch.append(dest_chan)
                               
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
            
            #destmask_batch=destmask_batch+dest_mask
            
        destmask_batch=torch.tensor(destmask_batch,dtype=torch.int32)
        hist_batch=torch.index_select(hist_batch, 1, destmask_batch)
        fut_batch=torch.index_select(fut_batch, 1, destmask_batch)
        
        mask_batch=torch.index_select(mask_batch, 0, destmask_batch)
        
        op_mask_batch=torch.index_select(op_mask_batch, 1, destmask_batch)
        
        mapfeats_batch=torch.index_select(mapfeats_batch, 0, destmask_batch)
        motiva_batch = torch.index_select(motiva_batch, 0, destmask_batch)        
        return hist_batch, nbrs_batch, mask_batch, fut_batch, op_mask_batch, mapfeats_batch, motiva_batch
    
    def collate_fn(self, samples):
        #print(len(samples[0][0]['fut']))
        #print(len(samples[1]))
        env1=samples[0][0]
        #print(len(samples[0][0]['labels']))
        env2=samples[0][1]
        hist_batch1, nbrs_batch1, mask_batch1, fut_batch1, op_mask_batch1, mapfeats_batch1, motiva_batch1=self.collate_samples(env1)
        hist_batch2, nbrs_batch2, mask_batch2, fut_batch2, op_mask_batch2, mapfeats_batch2,motiva_batch2=self.collate_samples(env2)
        
        traj1=hist_batch1, nbrs_batch1, mask_batch1, op_mask_batch1, mapfeats_batch1,motiva_batch1
        traj2=hist_batch2, nbrs_batch2, mask_batch2, op_mask_batch2, mapfeats_batch2,motiva_batch2
        
        env1={'traj':traj1,'fut':fut_batch1}
        env2={'traj':traj2,'fut':fut_batch2}
        
        return [env1,env2]
        
        

        
                    
   
