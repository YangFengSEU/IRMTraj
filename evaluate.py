from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
#from unidataset_highd import UniDataset
#from unidataset_mix import UniDataset
from unidataset_ngsim import UniDataset
#from unidataset_NandH import UniDataset
import time


## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 15
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False


# Evaluation metric:
metric = 'nll'  #or rmse


# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/new/cslstm_70.tar'))
if args['use_cuda']:
    net = net.cuda()

#tsSet = ngsimDataset('data/TestSet.mat')
#tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

tsSet=UniDataset('/home/lb/Documents/unidataset/processed/NGSIMnew/val/')
tsDataloader = DataLoader(tsSet,batch_size=1,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()

acc=0
k=0
for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, mask, lane_enc, fut, op_mask = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda().bool()
        lane_enc = lane_enc.cuda()
        #lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

    if metric == 'nll':
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lane_pred= net(hist, nbrs, mask, lane_enc)
            l,c = maskedNLLTest(fut_pred, lane_pred, fut, op_mask)
            acc+=(torch.sum(torch.max(lane_pred.data, 1)[1] == torch.max(lane_enc.data, 1)[1])).item() / lane_enc.size()[0]
            k=k+1
        else:
            fut_pred = net(hist, nbrs, mask, lane_enc)
            l, c = maskedNLLTest(fut_pred, 0, fut, op_mask,use_maneuvers=False)
            acc+=(torch.sum(torch.max(lane_pred.data, 1)[1] == torch.max(lane_enc.data, 1)[1])).item() / lane_enc.size()[0]
            k=k+1
    else:
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lane_pred= net(hist, nbrs, mask, lane_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lane_pred.shape[0]):
                lane_man = torch.argmax(lane_pred[k, :]).detach()
                #lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lane_man
                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
        else:
            fut_pred = net(hist, nbrs, mask, lane_enc)
            l, c = maskedMSETest(fut_pred, fut, op_mask)


    lossVals +=l.detach()
    counts += c.detach()

if metric == 'nll':
    print(lossVals / counts)
    print((acc/k)*100)
else:
    print(torch.pow(lossVals / counts,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters


