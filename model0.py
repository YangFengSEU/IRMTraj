from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utilsout import outputActivation
from backpack import backpack, extend
from backpack.extensions import BatchGrad


class decode(nn.Module):
    def __init__(self,args):
        super(decode, self).__init__()
        self.args = args
        self.out_length = args['out_length']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.use_maneuvers = args['use_maneuvers']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth*5
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.decoder_size =args['decoder_size']
        
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes*self.num_lon_classes, self.decoder_size,batch_first=True)
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size ,batch_first=True)
        self.op = torch.nn.Linear(self.decoder_size,2)
    
    def forward(self,enc):
        
        h_dec, _ = self.dec_lstm(enc)
                
        fut_pred = self.op(h_dec)
        print(2)

        return fut_pred


class highwayNet1(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(highwayNet1, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size =(((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth*5

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,3))
        self.soc_maxpool = torch.nn.MaxPool2d((2,2),padding = (1,1))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes*self.num_lon_classes, self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size ,batch_first=True)

        # Output layers:
        #self.op = torch.nn.Linear(self.decoder_size,2)
        
        self.op_lane = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes*self.num_lon_classes)
        #self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, )

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        '''
        mydecode = nn.Sequential(
            torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size,batch_first=True),
            ReduceTuple(index=0),
            torch.nn.Linear(self.decoder_size,2),
            )
        '''
        
        #self.decode=extend(decode(args),use_converter=True)
        self.decode = extend(torch.nn.Linear(self.decoder_size,2))

        
    def lossfun(self):
        return nn.MSELoss(reduction = 'sum')
    
    ## Forward Pass
    def forward(self,hist,nbrs,masks, op_mask,):

        ## Forward pass hist:
        print("hist_shape:{}".format(hist.size()))    
        hist=self.ip_emb(hist)
        print("linear_shape:{}".format(hist.size()))
        hist=self.leaky_relu(hist)
        print("relu:{}".format(hist.size()))
        _,(hist_enc,_) = self.enc_lstm(hist)
        print("hist_enc_shape:{}".format(hist_enc.size()))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        #print("nbrs_enc_shape:{}".format(nbrs_enc.size()))
        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        #print(soc_enc.size(),hist_enc.size())
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)

        #print("soc_enc1:{}".format(soc_enc.size()))
        soc_enc = soc_enc.permute(0,3,2,1)# change the dim
        #print(soc_enc.size(),hist_enc.size())
        

        ## Apply convolutional social pooling:
        #print("soc_enc2:{}".format(soc_enc.size()))
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        #print(soc_enc.size(),hist_enc.size())
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)

        ## Apply fc soc pooling

        enc = torch.cat((soc_enc,hist_enc),1)
        #print("enc_shape:{}".format(enc.size()))
        enc = enc.repeat(self.out_length, 1, 1).permute(1,0,2)
        h_dec, _ = self.dec_lstm(enc)
        #print(h_dec.shape)
        
                

        fut_pred = self.decode(h_dec)
        #print('inmodel',type(fut_pred))

        fut_pred = fut_pred*op_mask
        fut_pred = fut_pred.reshape(-1,1)
        return fut_pred, h_dec



class highwayNet3(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(highwayNet3, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size =(((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth*5
        self.mapsize=32
        
        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)#2 32

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)#32 64

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,3))
        self.soc_maxpool = torch.nn.MaxPool2d((2,2),padding = (1,1))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes*self.num_lon_classes, self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size+32, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)
        
        self.op_lane = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes*self.num_lon_classes)
        #self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, )

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        
        #map
        
        self.map_froemb=torch.nn.Linear(2,32)
        self.map_lstm = torch.nn.LSTM(32,128,1)
        
        self.map_midemb=torch.nn.Linear(128,64)
        
        self.map_lstm2 = torch.nn.LSTM(64,64,1)
        
        self.map_backemb=torch.nn.Linear(self.encoder_size,32)
        #self.map_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        #self.conv_3x1map = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,3))
        #self.map_maxpool = torch.nn.MaxPool2d((2,2),padding = (1,1))
        


    ## Forward Pass hist,nbrs,masks,lane_enc, op_mask
    def forward(self,hist,nbrs,masks, mapfeats):

        ## Forward pass hist:
        #print("hist_shape:{}".format(hist.size()))    
        hist=self.ip_emb(hist)
        #print("linear_shape:{}".format(hist.size()))
        hist=self.leaky_relu(hist)
        #print("relu:{}".format(hist.size()))
        _,(hist_enc,_) = self.enc_lstm(hist)
        #print("hist_enc_shape:{}".format(hist_enc.size()))
        '''
        hist_shape:torch.Size([15, 6, 2])
        linear_shape:torch.Size([15, 6, 32])
        relu:torch.Size([15, 6, 32])
        hist_enc_shape:torch.Size([1, 6, 64])

        '''
        #print("hist_enc_shape:{}".format(hist_enc.size()))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        #print("nbrs_enc_shape:{}".format(nbrs_enc.size()))
        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)# change te dim
        #print("enc_shape:{}".format(soc_enc.size()))

        
        
        soc_enc=self.soc_conv(soc_enc)
        soc_enc=self.leaky_relu(soc_enc)
        soc_enc=self.conv_3x1(soc_enc)
        soc_enc=self.leaky_relu(soc_enc)
        soc_enc = self.soc_maxpool(soc_enc)
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)

        #print("enc_shape:{}".format(soc_enc.size()))
        #print("hist_enc:{}".format(hist_enc.size()))
        
        mapfeature= mapfeats.view(hist.shape[1], 50,2)
        
        mapfeature=mapfeature.permute(1,0,2)
        #print(mapfeature.size())
        _, (map_enc,_) = self.enc_lstm(self.leaky_relu(self.map_froemb(mapfeature)))
        
        map_enc = self.leaky_relu(self.map_backemb(map_enc.view(map_enc.shape[1], map_enc.shape[2])))
        

        enc = torch.cat((soc_enc,hist_enc,map_enc),1)
        #print("enc_shape:{}".format(enc.size()))
        

        fut_pred = self.decode(enc)
        return fut_pred


    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred




