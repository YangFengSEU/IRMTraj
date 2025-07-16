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
        print(len(self.data_list))
        
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
                    a[maxid]=0
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
        fut = ori[:, self.hist_len:, 3:5] - ori[:, self.hist_len-1:self.hist_len, 3:5]

        
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
        R_list = []
        T_list = []
        for i in range(num):
            norm_v = hist[i, -1, :]-hist[i, 0, :]
            R_i =np.zeros([2,2])
            l = math.sqrt(norm_v[0]*norm_v[0]+norm_v[1]*norm_v[1])
            if l!=0.0:
                R_i[0][0] = norm_v[1]/l
                R_i[1][0] = -norm_v[0]/l
                R_i[0][1] = norm_v[0]/l
                R_i[1][1] = norm_v[1]/l
            else:
                R_i[0][0] = 1
                R_i[1][0] = 0
                R_i[0][1] = 0
                R_i[1][1] = 1          
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
        
        #计算临近车辆坐标，计算自车motivation
        nbrs = []
        motiva_list = []
        for i in range(num):
            grid_line = grid.reshape([num, -1])
            motiva = np.zeros([6,1])
            #print(fut[int(i)])
            fut_norm = fut[int(i)]@R_list[i]#+T_list[i]

            if (fut_norm[-1,0]-fut_norm[0,0])/((fut_norm[-1,1]-fut_norm[0,1])+0.000001)>0.5:
                motiva[2] = 1
            elif (fut_norm[-1,0]-fut_norm[0,0])/((fut_norm[-1,1]-fut_norm[0,1])+0.000001)<-0.5:
                motiva[0] = 1
            else:
                motiva[1] = 1
            v = fut_norm[1:,:]-fut_norm[:-1,:]
            if (v[-1,1]*v[-1,1]+v[-1,0]*v[-1,0])-(v[0,1]*v[0,1]+v[0,0]*v[0,0])>0.15:#
                motiva[3] = 1
            elif (v[-1,1]*v[-1,1]+v[-1,0]*v[-1,0])-(v[0,1]*v[0,1]+v[0,0]*v[0,0])<-0.1:
                motiva[5] = 1
            else:
                motiva[4] = 1
            motiva_list.append(motiva)

            nbr = []
            for j in grid_line[i]:
                if not j == -1:
                   # print(hist[int(i)].shape)
                   traj = hist[int(j)]@R_list[i]#
                   nbr.append(traj)
                else:
                    nbr.append(np.empty([0,2]))
            #nbr = np.stack(nbr)
            nbrs.append(nbr)
        
        '''
        nbrs = np.stack(nbrs)
 
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
                
                #lane.centerline=lane.centerline - end.reshape(-1, 2)
                lane.centerline=(lane.centerline + T_list[i])@R_list[i]
               
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

            mapfeats=np.array(mapfeats)
            mapfeats_list.append(mapfeats)
            
        #归一化为deltax，deltay
        hist=hist-ori[:, self.hist_len-1:self.hist_len, 3:5]
        #print(ori[0,:,3:5])
        hist_c = np.zeros_like(hist)
        fut_c = np.zeros_like(fut)
        for i in range(num):
            h = hist[int(i)]@R_list[i]

            
            f = fut[int(i)]@R_list[i]
            hist_c[i] = h
            fut_c [i] = f
            # hist_c[i,1:,:] = h[1:,:]-h[:-1,:]

            # fut_c[i,1:,:] = f[1:,:]-f[:-1,:]

        #print(ori[0,:,3:5])
        return hist_c, fut_c, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask, motiva_list
    
    def collate_fn(self, samples):
        veh_num = 0
        samplenum = 0
        maxmapnum=337
        for _, _, num,nbrs, _, _, _, _,_ in samples:
            for j in nbrs: 
                veh_num += sum([len(j[i])!=0 for i in range(len(j))])

            samplenum += num
        nbrs_batch = torch.zeros(self.hist_len, veh_num, 2)
        
        # Initialize 
        hist_batch = torch.zeros(self.hist_len, samplenum, 2)
        fut_batch = torch.zeros(self.fut_len, samplenum, 2)
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
        for hist, fut, num, nbrs,_ , op_mask, mapfeats_list, dest_mask , motiva in samples:
            
            for k in range(num):

                hist_batch[0:10, sample_count + k, :] = torch.from_numpy(hist[k].astype(float))
                fut_batch[0:15, sample_count + k, :] = torch.from_numpy(fut[k].astype(float))
                
                # print(op_mask.shape)
                op_mask_batch[0:15, sample_count + k,:] = torch.from_numpy(op_mask[k].astype(int))
                motiva_batch[sample_count + k,:] = torch.from_numpy(motiva[k][:,0].astype(float))
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
        mask_batch=torch.index_select(mask_batch, 0, destmask_batch)
        
        op_mask_batch=torch.index_select(op_mask_batch, 1, destmask_batch)
        
        mapfeats_batch=torch.index_select(mapfeats_batch, 0, destmask_batch)
        motiva_batch = torch.index_select(motiva_batch, 0, destmask_batch)

        #print("destmask_batch: ",destmask_batch,"end")
        #print("fut_batch: ",fut_batch[-1,:,:],"end")

            
        return hist_batch, nbrs_batch, mask_batch, fut_batch, op_mask_batch, mapfeats_batch, motiva_batch
                    
    
if __name__ == "__main__":

    batch_size = 1

    trSet1=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/FT3s/train/',city="FT")
    trSet2=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/OF3s/val/',city="OF")
    trSet3=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/EPR3s/train/',city="EP")
    
    trDataloader1 = DataLoader(trSet1,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=trSet1.collate_fn)
    trDataloader2 = DataLoader(trSet1,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=trSet2.collate_fn)
    trDataloader3 = DataLoader(trSet1,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=trSet3.collate_fn)
    
    import matplotlib.pyplot as plt
    import pickle as pkl
    print(len(trSet1))
    print(len(trSet2))
    print(len(trSet3))
    # for hist, fut, num,nbrs, _, _, _, _,motiva, ori  in trSet2:
    #     print(motiva)
    #     plt.plot(hist[0,:,0],hist[0,:,1],"r")
    #     plt.plot(fut[0,:,0],fut[0,:,1],"b")
    #     traj = ori[0,:, 3:5]

    #     h = traj[:10,:]-traj[9:10,:]
    #     f = traj[10:,:]-traj[9:10,:]

    #     plt.plot(h[:,0],h[:,1],"y")
    #     plt.plot(f[:,0],f[:,1],"k")
    #     plt.show()
    name = ["left_add","mid_add","right_add","left_still","mid_still","right_still","left_sub","mid_sub","right_sub"]
    cnt = 0
    for hist_c, fut_c, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask, motiva_list in trSet3:
        # print(num)
        # print(len(motiva_list))
        # print(len(hist_c))
        # print(len(fut_c))
        # print(len(nbrs))
        # print(len(vehtype))
        # print(len(op_mask))
        # print(len(mapfeats_list))
        # print(len(dest_mask))
        # print(dest_mask)
        for i in range(len(motiva_list)):
            if i not in dest_mask:
                continue
            data = [hist_c[i], fut_c[i], num, nbrs[i], vehtype[i], op_mask[i], mapfeats_list[i], motiva_list[i]]
            indice = np.where(motiva_list[i]==1.0)[0].tolist()
            # print(indice)
            #print(mapfeats_list[i])
            index = (indice[1]-3)*3+indice[0]
            save_path = f'/home/lb/Documents/unidataset/train_data/OF'
            path = os.path.join(save_path,name[index])
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path+'/'+str(cnt)+'.pkl','wb') as f:
                pkl.dump(data,f)
            cnt =cnt+1

             



   
