from __future__ import print_function
import torch


from model import highwayNet2
from utilsout import maskedNLL,maskedMSE,maskedNLLTest,maskedMSEpenalty,maskedMSEirm,maskedMSEdestpenalty,maskedMSEides
from torch.utils.data import DataLoader
from unidataset_IRMv1_3train import UniDataset
from unidataset_IRMv1_3 import UniDataset1
import torch.nn.functional as F
from collections import OrderedDict

from torch import nn, optim, autograd
import time
import math

import numpy as np
from my_utils import maskedTest
#from model import calcumse
from modellog import logger_init
import logging
from args import args
from eval import evalnet
import os

def penaltyZINdest(generated_dest, dest, dest_mask,maneuver,motiva, env_num):

    train_penalty = 0
    scale= torch.ones(dest.size()[0]).cuda().requires_grad_()
    muX = generated_dest[:,0]

    muY = generated_dest[:,1]
    x = dest[:, 0]
    y = dest[:, 1]
  
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    loss = torch.zeros_like(dest_mask)
    loss[:,0] = out*scale
    loss[:,1] = out*scale
    loss = loss*dest_mask
    loss = torch.sum(loss,dim = 1)


    env_loss = torch.zeros(env_num).cuda()
    for i in range(motiva.size()[0]):
        indice = torch.where(motiva[i]==1.0)[0].tolist()
        index = (indice[1]-3)*3+indice[0]
        env_loss[index] += loss[i]

    for i in range(env_num):
        grad = autograd.grad(
            env_loss[i],
            [scale],
            create_graph=True)[0]
        train_penalty += grad ** 2
    
    return torch.sum(train_penalty)/env_num

def penaltydest(generated_dest, dest, dest_mask):#claculate IRM loss by rmse,but the fun(grad) alway have sth wrong 
  #print(fut_pred.size())
  muX = generated_dest[:,0]
  #print(fut.shape)
  muY = generated_dest[:,1]
  x = dest[:, 0]
  y = dest[:, 1]
  
  out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
  #print(out.size())
  scale= torch.ones(dest.size()[0]).cuda().requires_grad_()
  
  loss = maskedMSEdestpenalty(out*scale, fut, dest_mask)

  grad = autograd.grad(loss, [scale], create_graph=True)[0]
  return torch.sum(grad**2)/(dest.size()[0])

timenow = time.strftime('%Y-%m-%d-%H:%M',time.localtime())

logger_init('/home/lb/Documents/IRM+socialpoolingmap+cvae-samecodeing+motivationeasy/conv-social-pooling-master/log',
            'log.txt',
            logging.INFO)

logging.info(f'algorithm: {args.algorithm}')
# Initialize network

if args.algorithm=='irm':
    net=highwayNet2(args)
elif args.algorithm=='Rex':
    net=highwayNet2(args)
    
if args.use_cuda:
    net = net.cuda()

checkpoint1=torch.load('/home/lb/Documents/IRM+socialpoolingmap+cvae-samecodeing+motivationeasy/trained_models/2024-04-09-22:42_EP/epoch_29.tar')

net.load_state_dict(checkpoint1)


pretrainEpochs = 1
trainEpochs = 30
lr=0.001

optimizer = torch.optim.Adam(net.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma = 0.8)
batch_size = 1
crossEnt = torch.nn.BCELoss()#er yuan jiaochashang qiu pingjun


## Initialize data loaders

# if args.test_city =="OF":
#     trSet=UniDataset('/home/lb/Documents/unidataset/lastversion/processed/FT3s/train/','/home/lb/Documents/unidataset/lastversion/processed/EPR3s/train/')
#     testSet=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/EP3s/val/',city="OF")
# else:
#     raise ValueError("Mismatch between test dataset and training dataset")


if args.test_city =="EP":
    trSet=UniDataset('/home/lb/Documents/unidataset/lastversion/processed/MA3s/train/','/home/lb/Documents/unidataset/lastversion/processed/GL3s/train/',city1="MA",city2="GL",)
    testSet=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/EP3s/val/',city="EP0")
else:
    raise ValueError("Mismatch between test dataset and training dataset")

trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=False,num_workers=16,collate_fn=trSet.collate_fn)
testDataloader = DataLoader(testSet,batch_size=batch_size,shuffle=False,num_workers=16,collate_fn=testSet.collate_fn)


## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf
g_i=0
best_ade3=math.inf
best_ade5=math.inf

#for epoch_num in range(pretrainEpochs+trainEpochs):
for epoch_num in range(1):
    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')


    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lane_acc= 0
    avg_lane_acc1 = 0


    avg_lane_acc2 = 0
    #avg_lon_acc = 0
    penaltyloss1=0
    penaltyloss2=0
    l1=0
    l2=0
    feature_list = []
    domian_labels = []
    maneuver_types = []
    for i, envs in enumerate(trDataloader):
    #en0+env1
        #data2=enumerate(trDataloader2)
        alculate_flag=True
        for edx, env in enumerate(envs):
            hist, nbrs, mask, op_mask, mapfeats, motiva= env['traj']
            st_time = time.time()
            mask = mask.bool()

            fut = env['fut']

            
            if args.use_cuda:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()

                fut = fut.cuda()
                op_mask = op_mask.cuda()
                mapfeats=mapfeats.cuda()
                motiva = motiva.cuda()
                
            dest=fut[-1,:,:]#要重新整理端点数据！


            dest_mask=torch.ones(dest.size()[0],2).cuda()
            dest_mask[torch.isnan(dest)]=0

            fut[torch.isnan(fut)] = 0
            dest[torch.isnan(dest)] = 0
            
            if torch.sum(dest)==0:
                alculate_flag=False #防止出现都没有终点的情况
                break
            
            if nbrs.shape[1] == 0:
                continue
               
            if args.algorithm=='irm':
                if not args.use_maneuvers:
                    
                    
                    
                    generated_dest, mu, logvar, fut_pred, maneuver , features = net.forward(hist, nbrs, mask, mapfeats, dest)
                    features = F.normalize(features, p=1, dim=1)

                    feature_list.append(features.cpu().detach().numpy())
                    maneuver_types.append(motiva.cpu().detach().numpy())
                    domian_labels.append(edx)

                    env['mseloss'] = maskedMSEirm(fut_pred, fut[:-1,:,:], op_mask[:-1,:,:])
                    #env['irm'] = penaltydest(generated_dest, dest, dest_mask)
                    env['irm'] = penaltyZINdest(generated_dest, dest, dest_mask, maneuver, motiva, args.env_num)
                    env['destloss'] = maskedMSEides(generated_dest, dest,dest_mask)
                    env['klloss'] = KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    #cls_weight = torch.tensor([0.166, 0.166, 0.166, 0.166, 0.166, 0.17]).cuda()

                    env['maneuver'] = crossEnt(maneuver,motiva)

                    if math.isnan(env['irm'].item()):
                        print("hist:",hist)
                        print("fut:",fut)
                        print("dest:",dest)
                        print("fut_pred",fut_pred)
                        print("generated_dest",generated_dest)
                        exit()        
                    #print("envnum",edx,"loss",env['mseloss'].item(),"| ",env['irm'].item(), "| ",env['destloss'].item(),"|",env['klloss'].item() )
            
            if args.algorithm=='ERM':
                if not args.use_maneuvers:
                    
                    
                    
                    generated_dest, mu, logvar, fut_pred, maneuver = net.forward(hist, nbrs, mask, mapfeats, dest)

                    env['mseloss'] = maskedMSEirm(fut_pred, fut[:-1,:,:], op_mask[:-1,:,:])
                    env['irm'] = penaltyZINdest(generated_dest, dest, dest_mask, maneuver, motiva, args.env_num)
                    #env['destloss'] = maskedMSEides(generated_dest, dest,dest_mask)
                    #print("dest:",dest)
                    #print("pre_dest:",generated_dest)
                    env['klloss'] = KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    #cls_weight = torch.tensor([0.166, 0.166, 0.166, 0.166, 0.166, 0.17]).cuda()

                    env['maneuver'] = crossEnt(maneuver,motiva)

             
            if args.algorithm=='Rex':
                if not args.use_maneuvers:
                    
                    generated_dest, mu, logvar, fut_pred = net.forward(hist, nbrs, mask, mapfeats, dest)
            
                    env['mseloss'] = maskedMSEirm(fut_pred, fut[:-1,:,:], op_mask[:-1,:,:])
                    env['destloss'] = maskedMSEides(generated_dest, dest,dest_mask)
                    #print("dest:",dest)
                    #print("pre_dest:",generated_dest)
                    env['klloss'] = KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
               
        if alculate_flag==False:
            continue
        if args.algorithm=='irm' and alculate_flag:
            train_mse = torch.tensor(0.).cuda()
            train_dest = torch.tensor(0.).cuda()
            train_kl = torch.tensor(0.).cuda()
            train_ssl = torch.tensor(0.).cuda()
            irm_penalty = torch.tensor(0.).cuda()
            for edx, env in enumerate(envs):

                train_mse=train_mse+env['mseloss']
                train_dest=train_dest+ env['destloss']
                train_kl=train_kl+ env['klloss']
                train_ssl=train_ssl+env['maneuver']
                irm_penalty = irm_penalty+env['irm']


            loss=train_mse.clone()/2+train_kl/2+train_dest/2+train_ssl/2
            
            weight_norm = torch.tensor(0.).cuda()
            plase=['encoder_dest',
                   'encoder_latent',
                   'dest_decoder']
            for name, w in net.named_parameters():
                if name in plase:
                    weight_norm += w.norm().pow(2)
    
            loss = loss+args.l2_regularizer_weight * weight_norm

            penalty_weight = (args.penalty_weight if i >= args.penalty_anneal_iters else 1.0)
            loss =loss+penalty_weight * irm_penalty/2
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss =loss/penalty_weight
            
        if args.algorithm =='Rex' and alculate_flag:
            train_mse=torch.stack([envs[0]['mseloss'], envs[1]['mseloss']]).mean()
            train_dest=torch.stack([envs[0]['destloss'], envs[1]['destloss']]).mean()
            train_kl=torch.stack([envs[0]['klloss'], envs[1]['klloss']]).mean()
            loss=train_mse.clone()
            loss+=train_kl+train_dest
            
            weight_norm = torch.tensor(0.).cuda()
            plase=['encoder_dest',
                   'encoder_latent',
                   'dest_decoder']
            for name, w in net.named_parameters():
                if name in plase:
                    weight_norm += w.norm().pow(2)
    
            loss = loss+args.l2_regularizer_weight * weight_norm
            rex_penalty = (envs[0]['mseloss'].mean() - envs[1]['mseloss'].mean())**2
            

            penalty_weight = (args.penalty_weight if i >= args.penalty_anneal_iters else 1.0)
            loss =loss+penalty_weight * rex_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss =loss/penalty_weight

        if args.algorithm =='ERM' and alculate_flag:
            train_mse=torch.stack([envs[0]['mseloss'], envs[1]['mseloss']]).mean()
            train_dest=torch.stack([envs[0]['destloss'], envs[1]['destloss']]).mean()
            train_kl=torch.stack([envs[0]['klloss'], envs[1]['klloss']]).mean()
            loss=train_mse.clone()
            loss+=train_kl+train_dest

        optimizer.zero_grad()
        loss.backward()
        g_i +=1
        
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        

        # all_features = np.concatenate(feature_list, axis=0)
        # all_maneuver_types = np.concatenate(maneuver_types, axis=0)
        # #all_domian_labels = np.concatenate(domian_labels)



        # np.save('all_features.npy',all_features)
        # np.save('all_maneuver_types.npy',all_maneuver_types)
        # np.save('all_domian_labels.npy',domian_labels)

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss+=loss.item()
        
        avg_tr_time += batch_time
        #print("7:{}".format(torch.cuda.memory_allocated(0)))
        if i%100 == 99:
            torch.cuda.empty_cache()
        if i%100 == 99:
            eta = avg_tr_time/100*((len(trSet))/batch_size-i)
            info_str = f' Epoch no : {epoch_num} | Epoch progress(%):{i/((len(trSet)*2)/batch_size)*100:.2f} | Avg train loss:{avg_tr_loss/100:.4f} | ETA(s):{int(eta)}'
            logging.info(info_str)
            avg_tr_loss = 0
            avg_lane_acc1 = 0

            avg_tr_time = 0

        # if i%3000 == 2999 :
            
        #     ade3 =evalnet(testDataloader,net,20)
        #     if ade3<best_ade3:
        #         best_ade3=ade3

        #         save_path = f'./trained_models/{timenow}_{args.test_city}/'
        #         file_name = f'epoch_{epoch_num}.tar'
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)
        #         torch.save(net.state_dict(), save_path+file_name)
        #     #if ade5<best_ade5:
        #         #best_ade5=ade5
        #     logging.info(f'Epoch no:{epoch_num}| best ade 3s: {best_ade3:.2f}')
        if i>20000:
            break
    
    all_features = np.concatenate(feature_list, axis=0)
    all_maneuver_types = np.concatenate(maneuver_types, axis=0)
    #all_domian_labels = np.concatenate(domian_labels)



    np.save('all_features.npy',all_features)
    np.save('all_maneuver_types.npy',all_maneuver_types)
    np.save('all_domian_labels.npy',domian_labels)
    scheduler.step()



