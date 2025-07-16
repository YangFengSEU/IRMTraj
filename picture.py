from __future__ import print_function
import torch


from model import highwayNet2
from model0 import highwayNet3
from utilsout import maskedNLL,maskedMSE,maskedNLLTest,maskedMSEpenalty,maskedMSEirm,maskedMSEdestpenalty,maskedMSEides
from torch.utils.data import DataLoader
from unidataset_IRMv1_3train import UniDataset
from unidataset_IRMv1_3_fortest import UniDataset1
#from unidataset_highd import UniDataset1
#from unidataset_NandH import UniDataset
from collections import OrderedDict
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim, autograd
import time
import math

from backpack import backpack, extend
from backpack.extensions import BatchGrad
import numpy as np
from my_utils import maskedTest
import matplotlib.pyplot as plt

import map_vis_without_lanelet
#from model import calcumse
      
  

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 10
args['out_length'] = 15
args['grid_size'] = (13,13)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3                                                                                                                                                                           
args['num_lon_classes'] = 2
args['use_maneuvers'] = False
args['train_flag'] = True

args['algorithm']='irm'
args['l2_regularizer_weight']=0.00110794568
args['penalty_anneal_iters']=100
args['penalty_weight']=2000.0

print(args['algorithm'])
writer=SummaryWriter()
# Initialize network


net1=highwayNet2(args)

net2=highwayNet3(args)
    
#writer.add_graph(net, (torch.randn(15, 11, 2),torch.randn(15, 72, 2),torch.zeros(11, 13, 13, 64),torch.randn(11, 6),torch.randn(337, 11, 2),torch.randn(337, 11, 2),torch.randn(337, 11)))

if args['use_cuda']:
    net1 = net1.cuda()
    net2 = net2.cuda()

checkpoint1=torch.load('/home/lb/Documents/IRM+map+cvae-samecodeing/conv-social-pooling-master/trained_models/intersection/FT+EP的VMIC.tar')
checkpoint2=torch.load('/home/lb/Documents/fishralpha1.0+map/conv-social-pooling-master/trained_models/intersection/FT+EP的ERM.tar')

net1.load_state_dict(checkpoint1)
net2.load_state_dict(checkpoint2)

batch_size = 1


testSet=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/OF3s/val/',city="OF")
#testSet=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/EP3s/val/',city="EP0")
testDataloader = DataLoader(testSet,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=testSet.collate_fn)



net1.eval()
net2.eval()



print(len(testDataloader))
for j, data in enumerate(testDataloader):
    
    if j>29:
        break

    hist, nbrs, mask, fut,origin, oritraj, op_mask, mapfeats= data
    
    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda().bool()

        
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        mapfeats=mapfeats.cuda()
   
     
    dest=fut[-1,:,:]


    dest_mask=torch.ones(dest.size()[0],2).cuda()
    dest_mask[torch.isnan(dest)]=0

    fut[torch.isnan(fut)] = 0
    dest[torch.isnan(dest)] = 0

    if nbrs.shape[1] == 0:
        continue
    

    all_l2_errors_dest = []
    all_guesses = []
    for _ in range(20):
        
        generated_dest,soc_enc,hist_enc,map_enc= net1.forward(hist, nbrs, mask, mapfeats)
        generated_dest=generated_dest.detach().cpu().numpy()
        all_guesses.append(generated_dest)
        destnum=dest.detach().cpu().numpy()
        l2error_sample = np.linalg.norm(generated_dest - destnum, axis = 1)
        all_l2_errors_dest.append(l2error_sample)
        
    all_l2_errors_dest = np.array(all_l2_errors_dest)
    all_guesses = np.array(all_guesses)
    
    # choosing the best guess
    indices = np.argmin(all_l2_errors_dest, axis = 0)
    #print(indices)
    #print(np.arange(hist.shape[0]))
    #print(np.arange(hist.shape[1]))
    best_guess_dest = all_guesses[indices,np.arange(hist.shape[1]),  :]
    # taking the minimum error out of all guess

    
    best_guess_dest = torch.Tensor(best_guess_dest).cuda()
    
    # using the best guess for interpolation
    fut_pred = net1.predict(hist, best_guess_dest, mask, soc_enc,hist_enc,map_enc)

    best_guess_dest=torch.unsqueeze(best_guess_dest, dim=0)
    #print(fut_pred.size())
    #print(best_guess_dest.size())        
    fut_predirm= net2(hist, nbrs, mask, mapfeats)
    predicted_future = torch.cat((fut_pred[:,:,0:2], best_guess_dest), axis = 0)
    predicted_future=predicted_future.detach().cpu()
    fut=fut.detach().cpu()
    hist=hist.detach().cpu()
    
    predicted_future=predicted_future+oritraj[9:10, :,:]
    #ori[:, self.hist_len:, 3:5] - ori[:, self.hist_len-1:self.hist_len, 3:5]
    fut=fut+oritraj[9:10, :,:]
    hist=hist+oritraj[9:10, :,:]
    
    
    fut_predirm=fut_predirm.detach().cpu()
    fut_predirm=fut_predirm[:,:,0:2]+oritraj[9:10, :,:]
    
    '''
    
    if j==86:

        for i in range(0,predicted_future.size()[1]):
            if  i==1 or i==2:
                continue
            plt.plot(predicted_future[:,i,0],predicted_future[:,i,1],"blue",zorder=30,linewidth=0.75)
            plt.plot(fut_predirm[:,i,0],fut_predirm[:,i,1],"green",zorder=20,linewidth=0.75)
            plt.plot(oritraj[:,i,0],oritraj[:,i,1],"black",zorder=15,linewidth=0.75)

      
    if j==5:
    
        for i in range(0,predicted_future.size()[1]):
            if i==0:
                    
                plt.plot(predicted_future[:,i,0],predicted_future[:,i,1],"blue",zorder=30,linewidth=0.75)
                plt.plot(fut_predirm[:,i,0],fut_predirm[:,i,1],"green",zorder=20,linewidth=0.75)
                plt.plot(oritraj[:,i,0],oritraj[:,i,1],"black",zorder=15,linewidth=0.75)

    if j==72:
    
        for i in range(0,predicted_future.size()[1]):
            if i==0 or i==1 or i==2 or i==4 or i==5 or i==7 or i==8 or i==9 :
                continue
            plt.plot(predicted_future[:,i,0],predicted_future[:,i,1],"blue",zorder=30,linewidth=0.75)
            plt.plot(fut_predirm[:,i,0],fut_predirm[:,i,1],"green",zorder=20,linewidth=0.75)
            plt.plot(oritraj[:,i,0],oritraj[:,i,1],"black",zorder=15,linewidth=0.75)      
    '''

    if j==28:
        fig, axes = plt.subplots(1, 1)
        lanelet_map_file='/home/lb/Downloads/interaction-dataset-master/interaction-dataset-master/maps/DR_DEU_Roundabout_OF.osm'
        #lanelet_map_file='/home/lb/Downloads/interaction-dataset-master/interaction-dataset-master/maps/DR_USA_Intersection_EP0.osm'
        lat_origin = 0.0
        lon_origin = 0.0
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)
        for i in range(0,predicted_future.size()[1]):
            if  i==1 or i==6 or i==2:
                continue
            plt.plot(predicted_future[:,i,0],predicted_future[:,i,1],"blue",zorder=30,linewidth=0.75)
            plt.plot(fut_predirm[:,i,0],fut_predirm[:,i,1],"green",zorder=20,linewidth=0.75)
            plt.plot(oritraj[:,i,0],oritraj[:,i,1],"black",zorder=15,linewidth=0.75)
        plt.show()
        
        path='./picture/picture'+str(j)+'.png'
        fig.savefig(path,dpi=300)
        
    #计算ade




           

                

       
 
