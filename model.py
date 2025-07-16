from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utilsout import outputActivation
#from swin_transformer import SwinTransformer
from torch.nn import functional as F
import time


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class highwayNet2(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(highwayNet2, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args.use_cuda

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args.use_maneuvers

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args.train_flag

        ## Sizes of network layers
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.in_length = args.in_length
        self.out_length = args.out_length
        self.grid_size = args.grid_size
        self.soc_conv_depth = args.soc_conv_depth
        self.conv_3x1_depth = args.conv_3x1_depth
        self.dyn_embedding_size = args.dyn_embedding_size
        self.input_embedding_size = args.input_embedding_size
        self.num_lat_classes = 3
        self.num_lon_classes = 2
        self.soc_embedding_size =(((args.grid_size[0]-4)+1)//2)*self.conv_3x1_depth*5
        self.mapsize=32
        self.zdim=16
        self.fdim=16
        self.sigma = 1.3
        
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
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size+self.mapsize+self.zdim, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)
        
        self.op_lane = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes*self.num_lon_classes)
        #self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, )

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        
        #map change embeding not with lstm
        
        
        self.map_froemb=torch.nn.Linear(2,32)
        self.map_lstm = torch.nn.LSTM(32,128,1)
        
        self.map_midemb=torch.nn.Linear(32*4,32)
        
        self.map_lstm2 = torch.nn.LSTM(64,64,1)
        
        self.map_backemb=torch.nn.Linear(64,32)
        
        self.encoder_past = MLP(input_dim = 32, output_dim = self.fdim, hidden_size=(512,256))
        
        self.encoder_dest = MLP(input_dim = 2, output_dim = self.fdim, hidden_size=(8,16))
        self.encoder_map = MLP(input_dim = 32, output_dim = self.fdim, hidden_size=(512,256))

        self.encoder_latent = MLP(input_dim = 3*self.fdim, output_dim = 2*self.zdim, hidden_size=(8,50))

        self.dest_decoder = MLP(input_dim = self.fdim + self.zdim, output_dim = 2, hidden_size=(1024,512,1024))
        
        self.map_conv = torch.nn.Conv2d(64,32,(5,4))
        #self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,3))
        self.map_maxpool = torch.nn.MaxPool2d((2,2),padding = (1,1))
        
        self.ssl_maneuver = torch.nn.Linear(400+32+32+16, 3)
        self.ssl_speed = torch.nn.Linear(400+32+32+16, 3)


    ## Forward Pass hist,nbrs,masks,lane_enc, op_mask
    def forward(self,hist,nbrs,masks, mapfeats, dest=None):
        start = time.perf_counter() 
        m_start = torch.cuda.memory_allocated()
        ## Forward pass hist:
        #print(hist.size())
        #print(masks.size())
        histemb=self.ip_emb(hist)
        histemb=self.leaky_relu(histemb)
        _,(hist_enc,_) = self.enc_lstm(histemb)
        
        '''
        hist_shape:torch.Size([15, 6, 2])
        linear_shape:torch.Size([15, 6, 32])
        relu:torch.Size([15, 6, 32])
        hist_enc_shape:torch.Size([1, 6, 64])

        '''
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))
        #print("_________________")
        #print(nbrs.size())

        ## Forward pass nbrs
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        #print(nbrs_enc.size())
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        #print(nbrs_enc.size())
        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()

        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)# change te dim
        #print(soc_enc.size())
        

        #print(soc_enc.size())
        soc_enc=self.soc_conv(soc_enc)
        soc_enc=self.leaky_relu(soc_enc)
        soc_enc=self.conv_3x1(soc_enc)
        soc_enc=self.leaky_relu(soc_enc)
        soc_enc = self.soc_maxpool(soc_enc)


        soc_enc = soc_enc.view(-1,self.soc_embedding_size)

        #print(mapfeats.size())
        mapfeature= mapfeats.view(hist.shape[1], 50,2)#8,50,2
        #print(mapfeature.size())
        hist=hist.permute(1,0,2)
        
        hist=hist.reshape(hist.size()[0],-1)
        ftraj = self.encoder_past(hist_enc)
        
        #print("*****************")

        #print(mapfeature.size())
        dt = time.perf_counter()
        m_dt = torch.cuda.memory_allocated()
        num=mapfeature.size()[0]
        _, (map_enc,_) = self.enc_lstm(self.leaky_relu(self.map_froemb(mapfeature)))
        #print(map_enc.size())
        map_enc=map_enc.repeat(num,1,1).view(-1,5,10,map_enc.size()[-1])
        #print(map_enc.size())
        

        map_enc = map_enc.permute(0,3,1,2)# change te dim
        #print(map_enc.size())
        map_enc=self.map_maxpool(self.leaky_relu(self.map_conv(map_enc)))
        #print(map_enc.size())
        map_enc = self.map_midemb(map_enc.reshape(-1,32*4))
        #print(map_enc.size())
        
        #print(a)
        #map_ori=mapfeats.reshape(mapfeats.size()[0],-1)
        mapvae=self.encoder_map(map_enc)
        dt_end = time.perf_counter()        
        m_dt_end = torch.cuda.memory_allocated()
        #print("dt",(dt_end-dt)*1000,m_dt_end-m_dt)
        if not self.training:
            z = torch.Tensor(hist.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points
            zdyc = time.perf_counter()
            m_zdyc = torch.cuda.memory_allocated()

            # CVAE code
            dest_features = self.encoder_dest(dest)
            features = torch.cat((ftraj, dest_features,mapvae), dim = 1)
            latent =  self.encoder_latent(features)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array

            var = logvar.mul(0.5).exp_() # mul是乘法的意思，然后exp_是求e的次方并修改原数值
            eps = torch.FloatTensor(var.size()).normal_() # 在cuda中生成一个std.size()的张量，标准正态分布采样，类型为FloatTensor
            eps = eps.cuda()
            z = eps.mul(var).add_(mu)

        z = z.double().cuda() #隐变量
        decoder_input = torch.cat((ftraj, z), dim = 1).float()
        generated_dest = self.dest_decoder(decoder_input)
        end =time.perf_counter()
        m_zdyc_end = torch.cuda.memory_allocated()
        #print("zdyc",(end-zdyc)*1000,m_zdyc_end-m_zdyc)
        #map_enc=self.map_emb(map_enc)
        
        #cvae map concat with traj
        
        #decoder concat with map and traj with IRM loss
        
        #decoder

        if self.training:
            generated_dest_features = self.encoder_dest(generated_dest)
            '''
            soc_enc:torch.Size([11, 400])
            hist_enc:torch.Size([11, 32])
            map_enc:torch.Size([11, 32])
            generated_dest_features:torch.Size([11, 16])
            '''
            #print(map_enc.shape)
            enc = torch.cat((soc_enc,hist_enc,map_enc,generated_dest_features),1)
            #print("enc_shape:{}".format(enc.size()))
            
            fut_pred = self.decode(enc)
            
            ###ssl enc
            motivation = time.perf_counter()
            m_motivation = torch.cuda.memory_allocated()
            maneuver = self.ssl_maneuver(enc)
            maneuver = self.softmax(maneuver)
            speed = self.ssl_speed(enc)
            speed = self.softmax(speed)
            maneuver = torch.cat((maneuver,speed),1)
            end = time.perf_counter()
            m_end = torch.cuda.memory_allocated()
            # print("motivation",(end-motivation)*1000,m_end-m_motivation)
            # print("ztmodel",(end-start)*1000,m_end-m_start)
            return generated_dest, mu, logvar, fut_pred, maneuver, enc
        return generated_dest,soc_enc,hist_enc,map_enc


    def decode(self,enc):
        '''
        enc_shape:torch.Size([24, 11, 480])
        h_dec:torch.Size([24, 11, 128])
        h_dec:torch.Size([11, 24, 128])
        fut_pred:torch.Size([11, 24, 5])
        fut_pred:torch.Size([24, 11, 5])
        fut_pred:torch.Size([24, 11, 5])
        '''
        enc = enc.repeat(self.out_length-1, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)  
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)      
        fut_pred = outputActivation(fut_pred)

        return fut_pred
    
    def predict(self, hist, generated_dest, mask, soc_enc,hist_enc,map_enc):

        generated_dest_features = self.encoder_dest(generated_dest)
        enc = torch.cat((soc_enc,hist_enc,map_enc,generated_dest_features),1)

        fut_pred = self.decode(enc)
        return fut_pred




