import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Blocks import BasicBlock, Bottleneck, ChannelChange


class Pure_Coding(nn.Module):
    def __init__(self, block = None, scale = 1, d = 15, filters = 50, num_m = 32, ds = False, trans = False):
        
        super(Pure_Coding, self).__init__()
        
#         self.filters = 50
        self.d = d
        self.num_m = num_m
        self.scale = scale
        self.filters = filters

        if block.__name__ == 'BasicBlock':
            layers = 4
        elif block.__name__ == 'Bottleneck':
            layers = 3
            
        self.means = torch.rand((self.d, self.num_m), device='cuda', requires_grad = True)
        
        self.initiated = False
        self.stage = 0

        print('run block')
            
        enc_layers = []
        enc_layers.append(block(1, self.filters))
        if ds == True:
            print('Down sampling')
            enc_layers.append(nn.Conv1d(self.filters, self.filters, 5, stride = 2, padding = 2))
            enc_layers.append(nn.ReLU())
#             layers -= 1
#         for i in range(layers-1):
        enc_layers.append(block(self.filters, self.filters))
        enc_layers.append(block(self.filters, self.filters))
        enc_layers.append(nn.Conv1d(self.filters, self.d, 5, padding=2))
#         enc_layers.append(block(self.d, self.d))
        self.enc = nn.Sequential(*enc_layers)
        
        
        trans_layer = nn.ConvTranspose1d(self.d, self.d, 5, stride = 2, padding = 2, \
                                         output_padding = 1)
        
        addlayers = []
        addlayers.append(nn.Conv1d(self.d, self.filters, 5, padding=2))
        for i in range(3):
            addlayers.append(block(self.filters, self.filters))
        self.addup_layers = nn.Sequential(*addlayers)
        
        
        dec_layers = []
#         dec_layers.append(block(self.d, self.d))
        dec_layers.append(nn.Conv1d(self.d, self.filters, 5, padding=2))
        
        if ds == True and trans == True:
            dec_layers.append(trans_layer)
#             layers -= 1
            print('Transconv')
        elif ds == True:
            print('Sub-pixel')
        for i in range(3):
            dec_layers.append(block(self.filters, self.filters))
        self.dec = nn.Sequential(*dec_layers)
        
        
        self.dec_sp1 = nn.Sequential(
#             block(self.d, self.d),
            nn.Conv1d(self.d, self.filters, 3, padding=1),
            block(self.filters, self.filters)
        )
        self.dec_sp2 = nn.Sequential(
            block(self.filters//2, self.filters//2),
            block(self.filters//2, self.filters//2)
        )
        
        dec_layers = []
        for i in range(layers):
            dec_layers.append(block(self.filters, self.filters))
        self.dec_full = nn.Sequential(*dec_layers)
    
        self.fc1 = nn.Linear(256 * self.filters, 256*self.filters//2)
        self.fc2 = nn.Linear(256*self.filters//2, 512)
        self.fc = nn.Linear(512*self.filters, 512)
        
        self.fc_sp = nn.Linear(512 * self.filters//2, 512)
        self.fc_full = nn.Linear(512 * self.filters, 512)
        
       
    def forward(self,x,soft): # Not using mask
        
        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  

        arg_idx = None
        
        if self.stage == 1:
            
            if self.initiated == False:
                self.mean = self.code_init(x, self.num_m)
                self.initiated = True
            
            x, arg_idx = self.code_assign(x, self.mean, soft = soft)    

        # Decoder    
        
        s_hat = self.dec(x) # -- shape (bs, d, 512)

        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat)) 
        return s_hat, arg_idx
    
    
    def forward_sp(self,x,soft): # Not using mask
        
        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  

        arg_idx = None
        
        if self.stage == 1:
            
            if self.initiated == False:
                self.mean = self.code_init(x, self.num_m)
                self.initiated = True
            
            x, arg_idx = self.code_assign(x, self.mean, soft = soft)    

        # Decoder    
        s_hat = self.dec_sp1(x) # -- shape (bs, d, 256)
        s_hat = self.sub_pixel(s_hat).cuda() # -- shape(bs, d//2, 512)
        s_hat = self.dec_sp2(s_hat) # -- shape (bs, d//2, 512)
        
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
#         print(s_hat.shape)
        s_hat = torch.tanh(self.fc_sp(s_hat)) 
        return s_hat, arg_idx
    
    def forward_full(self, x, soft):
        
        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  

        arg_idx = None
        
        if self.stage >= 1:
            
            if self.initiated == False:
                self.mean = self.code_init(x, self.num_m)
                self.initiated = True
            
            x, arg_idx = self.code_assign(x, self.mean, soft = soft)    
            
            if self.stage == 2:
                s_hat = self.addup_layers(x)
                s_hat = self.dec_full(s_hat) # -- shape (bs, d, 256)
                s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                s_hat = torch.tanh(self.fc_full(s_hat))

                return s_hat, arg_idx

        # Decoder    
        s_hat = self.dec(x) # -- shape (bs, d, 256)
        
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat))
        
        return s_hat, arg_idx
           
    
    def code_init(self, codes, num_m):
        
        idx = torch.randint(0, codes.shape[0] * codes.shape[-1], (num_m,))
        samples = codes.permute(0, 2, 1).reshape(-1, self.d)
        
        means = Variable(samples[idx].T, requires_grad = True)
        
        return means
        
    
    def code_assign(self, codes, mean, soft):
        
        # codes shape - (bs, d, L)
        # mean shape - (d, num_m)
        
        dist_mat = torch.zeros(codes.shape[0], codes.shape[-1], mean.shape[-1]).cuda()   # shape (bs, L, num_m)
        # Trade-off between computing speed and high-dimension matrix
        sec = 3
        step = codes.shape[0]//sec
        borders = torch.arange(0, codes.shape[0], step)

        if borders[-1] + step == codes.shape[0]:
            last = torch.Tensor([codes.shape[0]]).type(torch.int64)
            borders = torch.cat((borders, last), dim=0)
        else:
            borders[-1] = codes.shape[0]
            
        for i in range(sec):
            # batch shape (sec, d, L)
            batch = codes[borders[i]:borders[i+1]] 
            mat = torch.sub(batch[:, :, :, None], mean[None, :, None, :]) ** 2 # shape(sec, d, L, num_m)
            mat = torch.sum(mat, dim=1) # shape - (sec, L, num_m)
            dist_mat[borders[i]:borders[i+1]] = mat
#         mat = torch.sub(codes[:,:,:,None], mean[None,:,None,:]) ** 2

# #         # mat.shape(bs, d, 512, 32)
#         dist_mat = torch.sum(mat, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
            
        # Replace hidden features with means based on probability calculated by softmax

        arg_idx = None
        
        if soft == True:
            # Soft
            prob_mat = F.softmax(- dist_mat * self.scale, dim = -1) # shape(bs, 512, 32)
            x = torch.matmul(prob_mat, mean.transpose(0,1)) # shape(bs, 512, 10)
            x = x.permute(0, 2, 1)
            
            return x, prob_mat
        
        else:
            # Hard
            arg_idx = torch.argmax(-dist_mat, dim = -1) 
            # arg_idx.shape -> (bs, 512) entry is the index from 0-31
            x = mean[:, arg_idx]  # x.shape -> (10, bs, 512)
            x = x.permute(1, 0, 2)
        
        # arg_idx is only used for entropy calculating when doing hard argmax in test
            return x, arg_idx
    
       
    
    def sub_pixel(self, x):  
        
        # x.shape - (bs, self.d, L)
        # output.shape - (bs, self.d//2, L*2)
        
        bs = x.shape[0]
        d = x.shape[1] 
        L = x.shape[2]
        up_x = torch.zeros(bs, d//2, L*2)
        for i in range(0, d//2):
            x_sub = x[:, i*2:(i+1)*2, :] # (bs, 2, L)
            up_x[:,i,:] = x_sub.transpose(1,2).reshape(bs, 1, L*2)[:,0,:]
            
        return up_x
            
            
            
            
        
