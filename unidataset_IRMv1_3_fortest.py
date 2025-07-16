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

#from core.util.preprocessor.base import Preprocessor
#from cubic_spline import Spline2D



class UniDataset1:
    def __init__(self, folder, hist_len = 10, fut_len = 15 , enc_size=64, city="MA",obs_range=15):
        self.dir = folder
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.total_len = hist_len + fut_len
        
        self.enc_size=enc_size
        
        self.map=InterMap()
        self.data_list = os.listdir(self.dir)
        self.city=city
        self.obs_range = obs_range
    
    def __len__(self):
        return len(self.data_list) 
    
    def lane_candidate_sampling(self, centerline_list, distance=0.5, viz=False):
        """the input are list of lines, each line containing"""
        candidates = []
        for line in centerline_list:
            for i in range(len(line) - 1):
                if np.any(np.isnan(line[i])) or np.any(np.isnan(line[i+1])):
                    continue
                [x_diff, y_diff] = line[i+1] - line[i]#求datax和datay
                if x_diff == 0.0 and y_diff == 0.0:
                    continue
                candidates.append(line[i])

                # compute displacement along each coordinate
                den = np.hypot(x_diff, y_diff) + np.finfo(float).eps#返回直角三角形的边+上一个小量防止分母为0
                d_x = distance * (x_diff / den)
                d_y = distance * (y_diff / den)

                num_c = np.floor(den / distance).astype(np.int)
                pt = copy.deepcopy(line[i])
                for j in range(num_c):
                    pt += np.array([d_x, d_y])
                    candidates.append(copy.deepcopy(pt))
        candidates = np.unique(np.asarray(candidates), axis=0)#把车道等间距采样加密属于是


        return candidates
    
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
    
    def __getitem__(self, idx):
        
        ori = np.load(self.dir + self.data_list[idx], allow_pickle = True)
        
        hist = ori[:, :self.hist_len, 3:5]#3s-5s
        
        op_mask = ori[:, self.hist_len:, -1:]
        origin_fut=ori[:, self.hist_len:, 3:5]
        fut = ori[:, self.hist_len:, 3:5] - ori[:, self.hist_len-1:self.hist_len, 3:5]

        oritraj=ori[:, :, 3:5]
        vehtype=ori[:,0,4]
    
        num = hist.shape[0]
        
        dest=fut[:,-1,:]
        
        #res=0
        dest_mask = []
        for x in range(num):
            if not math.isnan(dest[x][0]):
                dest_mask.append(x)
        #print(dest_mask,"\n")
        #print(dest)

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
        for i in range(num):
            grid_line = grid.reshape([num, -1])
            
            nbr = []
            for j in grid_line[i]:
                if not j == -1:
                   # print(hist[int(i)].shape)
                   nbr.append(1*(hist[int(j)] - hist[int(i)][-1:, :]))
                else:
                    nbr.append(np.empty([0,2]))
            #nbr = np.stack(nbr)
            nbrs.append(nbr)
        
        #nbrs = np.stack(nbrs)
        '''
        ctr_line_candts_list=[]
        for i in range(num):
            
            end = hist[i].copy().astype(np.float32)
            #print(end)
            #print(type(end))
            #agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32)#agent未来轨迹
            ctr_line_candts = self.map.get_candidate_centerlines_for_traj(end, self.city)#获得车辆历史轨迹，根据最后一个点所在车道向周围扩展，找到可能的车道。
            #ctr_line_candts=np.concatenate(ctr_line_candts, 0)
            #print(ctr_line_candts)
            ctr_line_candts_list.append(ctr_line_candts)
            
            # rotate the center lines and find the reference center line
            #agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T#归一花后被旋转了
            #for i, _ in enumerate(ctr_line_candts):
                #ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T#把车道线也旋转
    
            tar_candts = self.lane_candidate_sampling(ctr_line_candts, viz=False)#把车道采样加密
            if self.split == "test":
                tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
                splines, ref_idx = None, None
            else:
                splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)#样条差值
                tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])#找出最接近未来轨迹的车道的01编号，并输出deltax，y        
            '''
        #########################
        #地图数据处理 需要：范围内车道，归一化feat，lane_idcs
        mapfeats_list=[]
        ctrs_list, feats_list, lane_idcs_list=[],[],[]
        tar_candts_list=[]
        maxlinelen=0
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        #print(num)
        for i in range(num):
            
            end = hist[i,-1].copy().astype(np.float32)
            lane_ids = self.map.get_lane_ids_in_xy_bbox(end[0], end[1], self.city, radius * 1)
            lane_ids = copy.deepcopy(lane_ids)
            lanes = dict()
            lane_distances=[]
            for lane_id in lane_ids:
                
                lane = self.map.city_lane_centerlines_dict[self.city][lane_id]
                lane = copy.deepcopy(lane)
                
                lane.centerline=lane.centerline - end.reshape(-1, 2)
               
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
                feat=np.asarray(ctrln[1:] - ctrln[:-1], np.float32)
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
        
          
        return hist, fut, origin_fut, oritraj, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask#, ctr_line_candts_list
        # return hist, fut, num, rad, traj, psi_rad
    
    def collate_fn(self, samples):
        veh_num = 0
        samplenum = 0
        maxmapnum=337
        for _, _,_, _,num,nbrs, _, _, _, _  in samples:
            for j in nbrs: 
                veh_num += sum([len(j[i])!=0 for i in range(len(j))])

            samplenum += num
        nbrs_batch = torch.zeros(self.hist_len, veh_num, 2)
        
        # Initialize 
        hist_batch = torch.zeros(self.hist_len, samplenum, 2)
        fut_batch = torch.zeros(self.fut_len, samplenum, 2)
        ori_fut_batch = torch.zeros(self.fut_len, samplenum, 2)
        ori_traj_batch = torch.zeros(self.fut_len+self.hist_len, samplenum, 2)
        op_mask_batch = torch.ones(self.fut_len, samplenum, 2)
        destmask_batch=[]
        
        pos=[0,0]
        mask_batch = torch.zeros(samplenum, 13, 13, self.enc_size)
        mask_batch = mask_batch.byte()
        
        mapfeats_batch=torch.zeros(samplenum,5,10,2)
        
        count = 0
        sample_count = 0
        dest_chan=0
        for hist, fut, origin_fut,ori_traj, num, nbrs,_ , op_mask, mapfeats_list, dest_mask in samples:
            
            for k in range(num):

                hist_batch[0:10, sample_count + k, :] = torch.from_numpy(hist[k].astype(float))
                fut_batch[0:15, sample_count + k, :] = torch.from_numpy(fut[k].astype(float))
                ori_fut_batch[0:15, sample_count + k, :] = torch.from_numpy(origin_fut[k].astype(float))
                ori_traj_batch[:, sample_count + k, :] = torch.from_numpy(ori_traj[k].astype(float))
                # print(op_mask.shape)
                op_mask_batch[0:15, sample_count + k,:] = torch.from_numpy(op_mask[k].astype(int))
                
                m=mapfeats_list[k].shape[0]
                if m>=5:
                    mapfeats_batch[sample_count + k,:,:,:]=torch.from_numpy(mapfeats_list[k][:5,:,:].astype(float))
                else:
                    mapfeats_batch[sample_count + k,0:m,:,:]=torch.from_numpy(mapfeats_list[k][:,:,:].astype(float))

            #print(dest_mask)    
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
        ori_fut_batch=torch.index_select(ori_fut_batch, 1, destmask_batch)
        ori_traj_batch=torch.index_select(ori_traj_batch, 1, destmask_batch)
        mask_batch=torch.index_select(mask_batch, 0, destmask_batch)
        
        op_mask_batch=torch.index_select(op_mask_batch, 1, destmask_batch)
        
        mapfeats_batch=torch.index_select(mapfeats_batch, 0, destmask_batch)
        

        #print("destmask_batch: ",destmask_batch,"end")
        #print("fut_batch: ",fut_batch[-1,:,:],"end")

            
        return hist_batch, nbrs_batch, mask_batch, fut_batch, ori_fut_batch,ori_traj_batch, op_mask_batch, mapfeats_batch
                    
    
if __name__ == "__main__":

    batch_size = 1

    trSet1=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/FT3s/train/',city="FT")
    trSet2=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/OF3s/val/',city="OF")
    trSet3=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/EPR3s/train/',city="EP")
    
    trDataloader1 = DataLoader(trSet1,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=trSet1.collate_fn)
    trDataloader2 = DataLoader(trSet1,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=trSet2.collate_fn)
    trDataloader3 = DataLoader(trSet1,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=trSet3.collate_fn)
    
    import matplotlib.pyplot as plt
    fig=plt.figure()
    al=fig.add_axes([0,0,1,1])
    a=trSet2[1]
    hist=a[0]
    traj=hist[7]
    print(traj)
    #plt.plot(traj[:,0:1],traj[:,1:2],"ob")
    ctr=a[6][7]
    for i in range(0,len(ctr)):
        
      plt.plot(ctr[i][:,0:1],ctr[i][:,1:2],"darkcyan")
    al.set_ylim(-20,25)
    al.set_xlim(-40,15)    
    plt.show()   
    '''
    for a in range(1000):
        hist=trSet3[a][0]
        num=hist.shape[0]
        for i in range(num):
            traj=hist[i]
            plt.plot(traj[:,0:1],traj[:,1:2],"b")
    


    trDataloader = DataLoader(trSet1,batch_size=1,shuffle=False,num_workers=8,collate_fn=trSet1.collate_fn)
    minn=100

    for i, data in enumerate(trSet1):
        hist, fut, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask=data
        

        for j in mapfeats_list:
            n=len(j)
            
            if minn>n:
                minn=n
            
    for i, data in enumerate(trSet2):
        hist, fut, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask=data
        

        for j in mapfeats_list:
            n=len(j)
            
            if minn>n:
                minn=n
        
    for i, data in enumerate(trSet3):
        hist, fut, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask=data
        

        for j in mapfeats_list:
            n=len(j)
            
            if minn>n:
                minn=n
    '''
    for j, data in enumerate(trDataloader1):
        hist, nbrs, mask,  fut, op_mask, mapfeat= data
        if j>2:
            break;
    '''    
    for j, data in enumerate(trDataloader2):
        hist, nbrs, mask,  fut, op_mask, mapfeat= data
        
    for j, data in enumerate(trDataloader3):
        hist, nbrs, mask,  fut, op_mask, mapfeat= data
    '''

   
