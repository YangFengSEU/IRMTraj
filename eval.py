import torch
from model import highwayNet2
from utilsout import maskedNLL,maskedMSE,maskedNLLTest,maskedMSEpenalty,maskedMSEirm,maskedMSEdestpenalty,maskedMSEides
from torch.utils.data import DataLoader
from unidataset_IRMv1_3train import UniDataset
from unidataset_IRMv1_3 import UniDataset1
#from unidataset_highd import UniDataset1
#from unidataset_NandH import UniDataset
from collections import OrderedDict

#from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim, autograd
import time
import math
import numpy as np
from my_utils import maskedTest
#from model import calcumse
import logging
from train_utils import get_optimizer,constrainScoreByWhole 
from args import args

def print_err(lossVals_r, counts, ade, scale, num):
    # ADE
    print('------------------ADE-----------------')

    ade3=math.inf
    ade5=math.inf
    count=0
    for j in range(4, num, 5):
        ade[count] = (torch.sum(lossVals_r[0:j+1]) / torch.sum(counts[0:j+1])) *scale

        print(ade[count])
        if count==2:
            ade3=ade[2]
        if count==4:
            ade5=ade[4]   
        count=count+1

    return ade3,ade5

def evalnet(testDataloader:DataLoader, net, best_of_n=1):
    ## test:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    net.eval()
    err_len = 1
    print('start eval. Calculating validation loss...')
    
    
    num = args.out_length
    lossVals = torch.zeros(err_len, num)
    lossVals_r = torch.zeros(err_len, num)
    counts = torch.zeros(err_len, num)

    for j, data in enumerate(testDataloader):
    

        hist, nbrs, mask, fut, op_mask, mapfeats, motiva= data
        
        # Initialize Variables
        if args.use_cuda:
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
        for _ in range(best_of_n):
            
            generated_dest,soc_enc,hist_enc,map_enc= net.forward(hist, nbrs, mask, mapfeats)
            generated_dest=generated_dest.detach().cpu().numpy()
            all_guesses.append(generated_dest)
            destnum=dest.detach().cpu().numpy()
            l2error_sample = np.linalg.norm(generated_dest - destnum, axis = 1)
            all_l2_errors_dest.append(l2error_sample)
            
        all_l2_errors_dest = np.array(all_l2_errors_dest)
        all_guesses = np.array(all_guesses)
        
        # choosing the best guess
        indices = np.argmin(all_l2_errors_dest, axis = 0)
        best_guess_dest = all_guesses[indices,np.arange(hist.shape[1]),  :]
        # taking the minimum error out of all guess
        best_guess_dest = torch.Tensor(best_guess_dest).cuda()
        
        # using the best guess for interpolation
        fut_pred = net.predict(hist, best_guess_dest, mask, soc_enc,hist_enc,map_enc)
        best_guess_dest=torch.unsqueeze(best_guess_dest, dim=0)    
        predicted_future = torch.cat((fut_pred[:,:,0:2], best_guess_dest), axis = 0)
        
        #计算ade

        for er in range(err_len):
                
            lane_pred = np.zeros([0, 6])
            #l1 = maskedMSE(fut_pred, fut, op_mask)
            l, l_r,c = maskedTest(predicted_future, lane_pred, fut, op_mask,use_maneuvers=False)
        
            lossVals[er]  += l.detach().cpu()
            lossVals_r[er]  += l_r.detach().cpu()
            counts[er]    += c.detach().cpu()

    ade3=math.inf
    ade=math.inf
    count=0
    scale = 0.85
    logging.info('-----fde------')
    fde = (lossVals_r[0] / counts) * scale
    fde_pr = []
    for j in range(4, num, 5):
        fde_pr.append(fde[j])
    logging.info(f"fde {fde_pr[0]},{fde_pr[1]},{fde_pr[2]}")
    logging.info('-----ade------')
    for m in range(4, num, 5):
        ade= (torch.sum(lossVals_r[0][0:m+1]) / torch.sum(counts[0][0:m+1])) *scale
        logging.info(str(ade.item()))
        if count==2:
            ade3=ade
  
        count=count+1
        
    net.train()
    return ade3       
  
        
